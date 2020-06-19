#vaemodel
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
from data_loader import Flickr30k as dataLoader
import final_classifier as  classifier
import models
from torchvision import models as torchModels
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import numpy as np

import matplotlib.pyplot as plt

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model,self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.attr = hyperparameters['attr']
        self.all_data_sources  = ['resnet_features', 'attributes']
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        #self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0]
        #self.att_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][1]
        #self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
       # self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        #self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source) , device= 'cuda')
        self.dataset = dataLoader(copy.deepcopy(self.auxiliary_data_source) , device= 'cuda')
        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_novel_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_novel_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_novel_classes = 10
        
        if self.attr == 'attributes':
            feature_dimensions = [2048, self.dataset.K]
        elif self.attr == 'bert':
            feature_dimensions = [2048, 768]

        # Here, the encoders and decoders for all modalities are created and put into dict
        
        self.feature_extractor = torchModels.resnet101(pretrained = True)
        
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, 2048)

        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources,feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)

            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())

        self.optimizer  = optim.Adam( parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label
    
    def trainstep(self, img, att):

        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        if self.attr == 'attributes':
            att_from_att[att_from_att < 0.5] = 0
            att_from_att[att_from_att > 0.5] = 1

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        if self.attr == 'attributes':
            att_from_img[att_from_img < 0.5] = 0
            att_from_img[att_from_img > 0.5] = 1

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        #if cross_reconstruction_loss>0:
        #    loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
           loss += distance_factor*distance

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_vae(self):

        losses = []

        self.dataloader = data.DataLoader(self.dataset,batch_size=self.batch_size,shuffle= True,drop_last=True)#,num_workers = 4)

        #self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        #self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()       
        
        #leave both statements
        self.train()
        self.reparameterize_with_noise = True
        
        metricsI1 = []
        metricsT1 = []
        metricsI5 = []
        metricsT5 = []
        
        self.feature_extractor.to(self.device)
        
        print('train for reconstruction')
        for epoch in range(0, self.nepoch ):
            self.current_epoch = epoch
            
            i=-1
            #for iters in range(0, len(self.dataset), self.batch_size):
            for iters in range(0, 150, self.batch_size):
                i+=1

                images, attributes, idxs = self.dataset.next_batch(self.batch_size, self.attr)
                
                               
                with torch.no_grad():
                    features = self.feature_extractor(images) 
                
                new_features = torch.zeros((5*features.size()[0],features.size()[1]))
                
                for z in range(0,features.size()[0]):
                    new_features[5*z:5*(z+1),:] = features[z,:].repeat(5,1)
                
                data_from_modalities = [new_features, attributes.type(torch.FloatTensor)]               
                
                
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(data_from_modalities[0], data_from_modalities[1] )

                if i%10==0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                    ' | loss ' +  str(loss))

                if i%50==0 and i>0:
                    losses.append(loss)
                
                images = images.cpu()
                idxs = idxs.cpu()
                
                attributes = attributes.cpu()
                
            for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].cpu()
            
            
            print('Generating gallery set...')
            self.generate_gallery()
            
            print('Generating t-SNE plot...')    
            z_imgs_embedded = TSNE(n_components=2).fit_transform(self.gallery_imgs_z.clone().cpu().detach())
            z_attrs_embedded = TSNE(n_components=2).fit_transform(self.gallery_attrs_z.clone().cpu().detach())
            
            
            plt.scatter(z_imgs_embedded[:,0], z_imgs_embedded[:,1], c = 'red')
            plt.scatter(z_attrs_embedded[:,0], z_attrs_embedded[:,1], c = 'blue')
            filename = 't-sne-plot-epoch'+str(epoch)+'.png'
            plt.savefig(filename)
            plt.clf()
            
            
            print('Evaluating retrieval...')
            metricsIepoch, metricsTepoch = self.retrieval()
            
            
            metricsI1.append(metricsIepoch[0])
            metricsT1.append(metricsTepoch[0])
            metricsI5.append(metricsIepoch[1])
            metricsT5.append(metricsTepoch[1])
            
            plt.plot(metricsI1, color = 'blue')
            plt.plot(metricsT1, color = 'red')
            plt.plot(metricsI5, color = 'orange')
            plt.plot(metricsT5, color = 'green')
            filename2 = 'recalls-plot-epoch'+str(epoch)+'.png'
            plt.savefig(filename2)
            plt.clf()
        
            print('Evaluation Metrics for image retrieval')
            print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsIepoch[0], metricsIepoch[1], metricsIepoch[2], metricsIepoch[3], metricsIepoch[4], metricsIepoch[5], metricsIepoch[6]))
            print('Evaluation Metrics for caption retrieval')
            print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsTepoch[0], metricsTepoch[1], metricsTepoch[2], metricsTepoch[3], metricsTepoch[4], metricsTepoch[5], metricsTepoch[6]))
            
            
               
        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses
    
    def retrieval(self):
                
        #nbrsI = NearestNeighbors(n_neighbors=self.dataset.ntest, algorithm='auto').fit(self.gallery_imgs_z.cpu().detach().numpy())
        nbrsI = NearestNeighbors(n_neighbors=150, algorithm='auto').fit(self.gallery_imgs_z.cpu().detach().numpy())
        #nbrsT = NearestNeighbors(n_neighbors=self.dataset.ntest, algorithm='auto').fit(self.gallery_attrs_z.cpu().detach().numpy())
        nbrsT = NearestNeighbors(n_neighbors=150, algorithm='auto').fit(self.gallery_attrs_z.cpu().detach().numpy())
        
        distI_dict = {}
        distT_dict = {}
        
        ranksI = np.zeros((1,5*self.dataset.ntest))
        ranksT = np.zeros((1,self.dataset.ntest))
        
        #for i in range(0, self.dataset.ntest):
        for i in range(0, 150):
            self.feature_extractor.to(self.device)
            
            image, attr, idx = self.dataset.get_item(i, self.attr)
            
                      
            with torch.no_grad():
                im_ft = self.feature_extractor(image) 
            
            mu_img, logvar_img = self.encoder['resnet_features'](im_ft)
            z_from_img = self.reparameterize(mu_img, logvar_img)

            mu_att, logvar_att = self.encoder['attributes'](attr.type(torch.FloatTensor).to(self.device))
            z_from_att = self.reparameterize(mu_att, logvar_att)
                     
            distancesI, indicesI = nbrsI.kneighbors(z_from_att.cpu().detach().numpy())      
            distancesT, indicesT = nbrsT.kneighbors(z_from_img.cpu().detach().numpy())
            
            distI_dict[i] = indicesI
            distT_dict[i] = indicesT
            
            
            for z in range(0,5):                   
                if len(np.where((indicesI[z] >= 5*i) & (indicesI[z] <= (5*i+4)))[0]) != 0:
                        ranksI[:,5*i + z] = np.where((indicesI[z] >= 5*i) & (indicesI[z] <= (5*i+4)))[0][0] 
                else:
                    ranksI[:,5*i + z] = 5000
            
            if any(map(len, np.where((indicesT[0] > (5*i-1)) & (indicesT[0] < (5*i+5))))):
                ranksT[:,i] = np.where((indicesT > (5*i-1)) & (indicesT < (5*i+5)))[1][0]
            else:
                ranksT[:,i] = 1000
            
            image = image.cpu()
            attr = attr.cpu()
        
        r1im = 100.0 * len(np.where(ranksI < 1)[1]) / len(ranksI[0,:])
        r5im = 100.0 * len(np.where(ranksI < 5)[1]) / len(ranksI[0,:])
        r10im = 100.0 * len(np.where(ranksI < 10)[1]) / len(ranksI[0,:])
        r50im = 100.0 * len(np.where(ranksI < 50)[1]) / len(ranksI[0,:])
        r100im = 100.0 * len(np.where(ranksI < 100)[1]) / len(ranksI[0,:])
        
        r1t = 100.0 * len(np.where(ranksT < 1)[1]) / len(ranksT[0,:])
        r5t = 100.0 * len(np.where(ranksT < 5)[1]) / len(ranksT[0,:])
        r10t = 100.0 * len(np.where(ranksT < 10)[1]) / len(ranksT[0,:])
        r50t = 100.0 * len(np.where(ranksT < 50)[1]) / len(ranksT[0,:])
        r100t = 100.0 * len(np.where(ranksT < 100)[1]) / len(ranksT[0,:])
        
        medrI = np.floor(np.median(ranksI)) + 1
        meanrI = ranksI.mean() + 1
        
        medrT = np.floor(np.median(ranksT)) + 1
        meanrT = ranksT.mean() + 1
        
        metricsI = [r1im, r5im, r10im, r50im, r100im, medrI, meanrI]
        metricsT = [r1t, r5t, r10t, r50t, r100t, medrT, meanrT]
        
        return metricsI, metricsT
        
    def generate_gallery(self):
              
        self.feature_extractor.to(self.device)
              
        z_imgs = []
        z_attrs = []
        rec_imgs = []
        rec_attrs = []
        
        y = 0
        i=-1
        #for iters in range(0, self.dataset.ntest, 50):
        for iters in range(0, 150, 50):
            i+=1
            
            images, attributes, idxs = self.dataset.next_batch_test(50, y, self.attr)
            idxs = idxs.cpu()
            
            with torch.no_grad():
                features = self.feature_extractor(images)
                
            data_from_modalities = [features, attributes.type(torch.FloatTensor)]
            
            images = images.cpu()

            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(self.device)
                data_from_modalities[j].requires_grad = False
            
            mu_img, logvar_img = self.encoder['resnet_features'](data_from_modalities[0])
            z_from_img = self.reparameterize(mu_img, logvar_img)
            
            mu_att, logvar_att = self.encoder['attributes'](data_from_modalities[1])
            z_from_att = self.reparameterize(mu_att, logvar_att)
                        
            
            if y == 0:
                z_imgs = z_from_img.cpu()
                z_attrs = z_from_att.cpu()
            else:
                z_imgs = torch.cat((z_imgs.cpu(),z_from_img.cpu()), dim = 0).cpu()
                z_attrs = torch.cat((z_attrs.cpu(),z_from_att.cpu()), dim = 0).cpu()
                
            
            y = y + 1
            
            print('iter: '+str(iters))
        
               
        self.gallery_imgs_z = z_imgs.cpu()
        print(self.gallery_imgs_z.size())
        self.gallery_attrs_z = z_attrs.cpu()
        print(self.gallery_attrs_z.size())
        
             
        
        
    def train_classifier(self, show_plots=False):

        if self.num_shots > 0 :
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        history = []  # stores accuracies


        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses


        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']


        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)

            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)

            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)


        if self.generalized:
            print('mode: gzsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            print('mode: zsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)


        clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)

            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples )

            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples )

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    novelclass_aux_data,
                    novel_corresponding_labels,self.att_unseen_samples )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data,
                seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

            train_Z = [z_seen_img, z_unseen_img ,z_seen_att    ,z_unseen_att]
            train_L = [img_seen_label    , img_unseen_label,att_seen_label,att_unseen_label]

            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################

        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)

        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:

                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

                history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
                                torch.tensor(cls.H).item()])

            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, torch.tensor(cls.acc).item(), 0])

        if self.generalized:
            return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(
                cls.H).item(), history
        else:
            return 0, torch.tensor(cls.acc).item(), 0, history
