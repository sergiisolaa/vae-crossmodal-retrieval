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
from dataloader import Flickr30k as dataLoader
import final_classifier as  classifier
import models
from torchvision import models as torchModels
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial import distance
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
        self.margin = hyperparameters['margin_loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        #self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source) , device= 'cuda')
        self.dataset = dataLoader(copy.deepcopy(self.auxiliary_data_source) , device= 'cuda', attr = self.attr)
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
            feature_dimensions = [2048, 768] #2048, 768

        # Here, the encoders and decoders for all modalities are created and put into dict
        
        self.fc_ft = nn.Linear(2048,2048)
        self.fc_ft.to(self.device)
        
        self.ft_bn = nn.BatchNorm1d(2048).to(self.device)
        
        self.fc_at = nn.Linear(self.dataset.K, self.dataset.K)
        self.fc_at.to(self.device)
        self.at_bn = nn.BatchNorm1d(self.dataset.K).to(self.device)

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
        parameters_to_optimize += list(self.fc_ft.parameters())
        parameters_to_optimize += list(self.fc_at.parameters())
        parameters_to_optimize += list(self.ft_bn.parameters())
        parameters_to_optimize += list(self.at_bn.parameters())
        
        
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
        img_in = F.normalize(self.rmac(img), p=2, dim = 1).squeeze(-1).squeeze(-1)
        img_in = F.normalize(img, p=2, dim=1)
        img_in = self.ft_bn(img_in)
        
        img_in = self.fc_ft(img_in)
        
        att_in = F.normalize(att, p=2, dim=1)
        att_in = self.at_bn(att_in)
        
        att_in = self.fc_at(att_in)   
        
        #Add non-linearity?
        
        
        mu_img, logvar_img = self.encoder['resnet_features'](img_in)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att_in)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        
        
        

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)    

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        
        
        
        
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
              
        #distanceI = torch.sum((mu_img - mu_att) ** 2, dim=1) 
        #distanceT = torch.sum((mu_att - mu_img) ** 2, dim=1)
        
        #tripletsI = []
        #tripletsT = []
        
        #for i in range(0, mu_img.shape[0]):
        #    for j in range(0, mu_img.shape[0]):
        #        if i != j:
        #            distI = distanceI[i] - torch.sum((mu_img[i] - mu_att[j]) ** 2) + self.margin
        #            distI = torch.max(torch.FloatTensor([0]), distI)
        #            distI += torch.sum((torch.sqrt(logvar_img[i].exp()) - torch.sqrt(logvar_att[i].exp())) ** 2, dim=0)
        #            distI = torch.sqrt(distI)
        #            tripletsI.append(distI)
                    
        #            distT = distanceT[i] - torch.sum((mu_att[i] - mu_img[j]) ** 2) + self.margin
        #            distT = torch.max(torch.FloatTensor([0]), distT)
        #            distT += torch.sum((torch.sqrt(logvar_att[i].exp()) - torch.sqrt(logvar_img[i].exp())) ** 2, dim=0)
        #            distT = torch.sqrt(distT)
        #            tripletsT.append(distT)
                    
                        
        
        #tripletI2t = sum(tripletsI)
        #tripletT2i = sum(tripletsT)
        
        #distance = tripletI2t + tripletT2i
        
        #distance = torch.FloatTensor([distance]).to(self.device)

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

        if cross_reconstruction_loss>0:
           loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
           loss += distance_factor*distance

        loss.backward()

        self.optimizer.step()

        return loss.item(), reconstruction_loss, beta*KLD, cross_reconstruction_factor*cross_reconstruction_loss, distance_factor*distance

    def train_vae(self):

        elosses = []
        lossesR = []
        lossesK = []
        lossesC = []
        lossesD = []

        self.dataloader = data.DataLoader(self.dataset,batch_size=self.batch_size,shuffle= True,drop_last=True)#,num_workers = 4)

        #self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        #self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()       
        
        #leave both statements
        self.train()
        self.fc_ft.train()
        self.fc_at.train()
        self.reparameterize_with_noise = True
        
        metricsI = []
        metricsT = []
        
        
        print('train for reconstruction')
        for epoch in range(0, self.nepoch ):
            self.train()
            
            self.current_epoch = epoch
            
            losses = []
            ilossesR = []
            ilossesK = []
            ilossesC = []
            ilossesD = []
            
            i=-1
            y = 0
            for iters in range(0, len(self.dataset), self.batch_size):
            
            #for iters in range(0, 1000, self.batch_size):
                i+=1

                features, attributes, idxs = self.dataset.next_batch(self.batch_size) #Si no és test treure la y
                
                
                data_from_modalities = [features, attributes.type(torch.FloatTensor)]               
                
                
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False

                loss, lossR, lossK, lossC, lossD = self.trainstep(data_from_modalities[0], data_from_modalities[1] )

                if i%10==0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                    ' | loss ' +  str(loss))

                losses.append(loss)                
                ilossesR.append(lossR)
                ilossesK.append(lossK)
                ilossesC.append(lossC)
                ilossesD.append(lossD)
                
                idxs = idxs.cpu()
                
                attributes = attributes.cpu()
                
                y += 1
                
            y = 0
            
            mean_loss = sum(losses)/len(losses)            
            elosses.append(mean_loss)
            print('epoch ' + str(epoch) + 'Loss: ' + str(loss))
            
            lossesR.append(sum(ilossesR)/len(ilossesR))
            lossesK.append(sum(ilossesK)/len(ilossesK))
            lossesC.append(sum(ilossesC)/len(ilossesC))
            lossesD.append(sum(ilossesD)/len(ilossesD))
            
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
            metricsI.append([metricsIepoch[0], metricsIepoch[1], metricsIepoch[2]])
            metricsT.append([metricsTepoch[0], metricsTepoch[1], metricsTepoch[2]])
        
            print('Evaluation Metrics for image retrieval')
            print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsIepoch[0], metricsIepoch[1], metricsIepoch[2], metricsIepoch[3], metricsIepoch[4], metricsIepoch[5], metricsIepoch[6]))
            print('Evaluation Metrics for caption retrieval')
            print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsTepoch[0], metricsTepoch[1], metricsTepoch[2], metricsTepoch[3], metricsTepoch[4], metricsTepoch[5], metricsTepoch[6]))
        
               
        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        
        
        import os

        file_name = "losses-DA.png"
        file_name2 = 'metrics-DA.png'
        if os.path.isfile(file_name):
            expand = 1
            while True:
                expand += 1
                new_file_name = file_name.split(".png")[0] + str(expand) + ".png"
                new_file_name2 = file_name2.split('.png')[0] + str(expand) + '.png'
                if os.path.isfile(new_file_name):
                    continue
                else:
                    file_name = new_file_name
                    file_name2 = new_file_name2
                    break
                
        #Plot de les losses i desar-lo
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(np.arange(self.nepoch), elosses, label="Total loss")
        plt.plot(np.arange(self.nepoch), lossesR, label = 'Reconstruction loss')
        plt.plot(np.arange(self.nepoch), lossesK, label = 'KL Divergence loss')
        plt.plot(np.arange(self.nepoch), lossesC, label = 'Cross-Reconstruction loss')
        plt.plot(np.arange(self.nepoch), lossesD, label = 'Distance-Alignment loss')
        plt.legend()
        plt.show()
        plt.savefig(file_name)
        plt.clf()
        
        #Plot de les metrics
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.plot(np.arange(self.nepoch), metricsI[0], label = 'T2I R@1')
        plt.plot(np.arange(self.nepoch), metricsI[1], label = 'T2I R@5')
        plt.plot(np.arange(self.nepoch), metricsT[0], label = 'I2T R@1')
        plt.plot(np.arange(self.nepoch), metricsT[1], label = 'I2T R@5')
        plt.legend()
        plt.show()
        plt.savefig(file_name2)
        plt.clf()
                
        return losses, metricsI, metricsT
    
    
    def retrieval(self):
        
        self.eval()
        
        def lda(self, x, y):
            distance = torch.sqrt(torch.sum((x[0] - y[0]) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(x[1].exp()) - torch.sqrt(y[1].exp())) ** 2, dim=1))
            
            return distance
        
        #nbrsI = NearestNeighbors(n_neighbors=self.dataset.ntest, algorithm='auto').fit(self.gallery_imgs_z.cpu().detach().numpy())
        #nbrsI = NearestNeighbors(n_neighbors=1000, algorithm='auto').fit(self.gallery_imgs_z.cpu().detach().numpy())
        #nbrsT = NearestNeighbors(n_neighbors=self.dataset.ntest, algorithm='auto').fit(self.gallery_attrs_z.cpu().detach().numpy())
        #nbrsT = NearestNeighbors(n_neighbors=5000, algorithm='auto').fit(self.gallery_attrs_z.cpu().detach().numpy())
        
        distI_dict = {}
        distT_dict = {}
        
        ranksI = np.zeros((1,5*self.dataset.ntest))
        ranksT = np.zeros((1,self.dataset.ntest))
        
        for i in range(0, self.dataset.ntest):
        #for i in range(0, 500):
            
            mu_img = self.gallery_imgs_z[i,:].unsqueeze(0)  
            mu_att = self.gallery_attrs_z[5*i:5*i + 5,:]
            
            #Add L2 norm and BatchNorm?
            #im_ft = F.normalize(im_ft, p=2, dim = 1)
            #im_ft = self.ft_bn(im_ft)
        
            #im_ft = self.fc_ft(im_ft)
            
        
            #attr = F.normalize(attr.type(torch.FloatTensor).to(self.device), p=2, dim=1)
            #attr = self.at_bn(attr)
        
            #attr = self.fc_at(attr)  
            
            #mu_img, logvar_img = self.encoder['resnet_features'](im_ft)
            #z_from_img = self.reparameterize(mu_img, logvar_img)

            #mu_att, logvar_att = self.encoder['attributes'](attr.type(torch.FloatTensor).to(self.device))
            #z_from_att = self.reparameterize(mu_att, logvar_att)
            
            #img = [mu_img.cpu().detach().numpy(), logvar_img.cpu().detach().numpy()]
            #att = [mu_att.cpu().detach().numpy(), logvar_att.cpu().detach().numpy()]
            
            distancesI = distance.cdist(mu_att.cpu().detach().numpy(), self.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
            distancesT = distance.cdist(mu_img.cpu().detach().numpy(), self.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
            
            indicesI = np.argsort(distancesI)
            indicesT = np.argsort(distancesT[0,:])
            
            for z in range(0,5):
                if len(indicesI[z] == i) != 0:
                    ranksI[:,(5*i) + z] = np.where(indicesI[z] == i)[0][0]
                else:
                    ranksI[:,(5*i) + z] = 1000
            
            
            if len(np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))) != 0:
                ranksT[:,i] = np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))[0][0]
            else:
                ranksT[:,i] = 1000
            
            '''
            for z in range(0,5):                   
                if len(np.where((indicesI[z] >= 5*i) & (indicesI[z] <= (5*i+4)))[0]) != 0:
                        ranksI[:,5*i + z] = np.where((indicesI[z] >= 5*i) & (indicesI[z] <= (5*i+4)))[0][0] 
                else:
                    ranksI[:,5*i + z] = 5000
            
            if len(np.where((indicesT[0] > (5*i-1)) & (indicesT[0] < (5*i+5)))) != 0:
                ranksT[:,i] = np.where((indicesT[0] > (5*i-1)) & (indicesT[0] < (5*i+5)))[0][0]
            else:
                ranksT[:,i] = 1000
            
            '''
            '''
            for z in range(0,5):                   
                if len(np.where((indicesT[z] >= 5*i) & (indicesT[z] <= (5*i+4)))[0]) != 0:
                        ranksT[:,5*i + z] = np.where((indicesT[z] >= 5*i) & (indicesT[z] <= (5*i+4)))[0][0] 
                else:
                    ranksT[:,5*i + z] = 5000
            
            if len(np.where(indicesI[0] == i)) != 0:
                ranksI[:,i] = np.where(indicesI[0] == i)[0][0]
            else:
                ranksI[:,i] = 1000
            
            '''            
            
        
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
        
        self.eval()
        
        z_imgs = []
        z_vars_im = []
        z_attrs = []
        z_vars_att = []
        rec_imgs = []
        rec_attrs = []
        
        y = 0
        i=-1
        for iters in range(0, self.dataset.ntest, 50):
        #for iters in range(0, 500, 50):
            i+=1
            
            features, attributes, idxs = self.dataset.next_batch_test(50, y)
            idxs = idxs.cpu()
              
            
            data_from_modalities = [features, attributes.type(torch.FloatTensor)]

            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(self.device)
                data_from_modalities[j].requires_grad = False
                if j== 0:
                    data_from_modalities[j] = F.normalize(self.rmac(data_from_modalities[j]), p=2, dim = 1).squeeze(-1).squeeze(-1)
                    data_from_modalities[j] = F.normalize(data_from_modalities[j], p=2, dim=1)
                    data_from_modalities[j] = self.ft_bn(data_from_modalities[j])
                    data_from_modalities[j] = self.fc_ft(data_from_modalities[j])
                elif j == 1: 
                    #Add L2 norm and BatchNorm?
                    data_from_modalities[j] = F.normalize(data_from_modalities[j], p=2, dim=1)
                    data_from_modalities[j] = self.at_bn(data_from_modalities[j])
                    data_from_modalities[j] = self.fc_at(data_from_modalities[j])
            
            mu_img, logvar_img = self.encoder['resnet_features'](data_from_modalities[0])
            z_from_img = self.reparameterize(mu_img, logvar_img)
            
            mu_att, logvar_att = self.encoder['attributes'](data_from_modalities[1])
            z_from_att = self.reparameterize(mu_att, logvar_att)
                        
            
            if y == 0:
                z_imgs = z_from_img.cpu()
                z_vars_im = logvar_img.cpu()
                z_attrs = z_from_att.cpu()
                z_vars_att = logvar_att.cpu()
            else:
                z_imgs = torch.cat((z_imgs.cpu(),z_from_img.cpu()), dim = 0).cpu()
                z_vars_im = torch.cat((z_vars_im.cpu(),logvar_img.cpu()), dim = 0).cpu()
                z_attrs = torch.cat((z_attrs.cpu(),z_from_att.cpu()), dim = 0).cpu()
                z_vars_att = torch.cat((z_vars_att.cpu(),logvar_att.cpu()), dim = 0).cpu()
                
            
            y = y + 1
            
            print('iter: '+str(iters))
        
               
        self.gallery_imgs_z = z_imgs.cpu()
        self.gallery_vars_im = z_vars_im.cpu()
        print(self.gallery_imgs_z.size())
        self.gallery_attrs_z = z_attrs.cpu()
        self.gallery_vars_att = z_vars_att.cpu()
        print(self.gallery_attrs_z.size())  
    
    def obtain_embeds(self,vec, modality = 'image'):
        
        if modality == 'image':
            vec = F.normalize(self.rmac(vec), p=2, dim = 1).squeeze(-1).squeeze(-1)
            vec = F.normalize(vec, p=2, dim=1)
            vec = self.ft_bn(vec)
            vec = self.fc_ft(vec)
            
            mu, logvar = self.encoder['resnet_features'](vec)
            z = self.reparameterize(mu, logvar)
        
        elif modality == 'attributes':
            vec = F.normalize(vec, p=2, dim=1)
            vec = self.at_bn(vec)
            vec = self.fc_at(vec)
            
            mu, logvar = self.encoder['attributes'](vec)
            z = self.reparameterize(mu, logvar)
        
        return mu, logvar, z
        
    #####################################################################################################################
    #Function extracted from GitHub: 
    #
    #filipradenovic/cnnimageretrieval-pytorch 
    #(https://github.com/filipradenovic/cnnimageretrieval-pytorch/)
    def rmac(self,x, L=3, eps=1e-6):
        ovr = 0.4 # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension
    
        W = x.size(3)
        H = x.size(2)
    
        w = min(W, H)
        w2 = math.floor(w/2.0 - 1)
    
        b = (max(H, W)-w)/(steps-1)
        (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension
    
        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:  
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1
    
        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)
    
        for l in range(1, L+1):
            wl = math.floor(2*w/(l+1))
            wl2 = math.floor(wl/2 - 1)
    
            if l+Wd == 1:
                b = 0
            else:
                b = (W-wl)/(l+Wd-1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
            if l+Hd == 1:
                b = 0
            else:
                b = (H-wl)/(l+Hd-1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
                
            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                    R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                    v += vt
    
        return v
        
        
    
