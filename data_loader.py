import numpy as np
import scipy.io as sio
import torch
from torchvision import transforms
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
import json
from PIL import Image
from vocabulary import VocabularyTokens
from transformers import BertTokenizer, BertModel

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class Flickr30k(object):
    def __init__(self, path, device = 'cuda', imagesize = 224):
        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder
        
        self.path = os.path.join(project_directory,'data','flickr30k')
        
        self.image_path = os.path.join(self.path,'images')
        print('The images folder is')
        print(self.image_path)
        
        self.device = device
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
        #self.K = 50
        self.T = 150
        
        
        self.imagesize = imagesize
        
        self.transforms = transforms.Compose([ 
            transforms.Resize(self.imagesize),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.train_imgs = []
        self.train_imgs_id = []
        self.train_sentences = {}
        self.train_sents_ids = {}
        
        self.val_imgs = []
        self.val_imgs_id = []
        self.val_sentences = {}
        self.val_sents_ids = {}
        
        self.test_imgs = []
        self.test_imgs_id = []
        self.test_sentences = {}
        self.test_sents_ids = {}
        
        print(self.path)
        attr_filename = os.path.join(self.path,'dataset.json')
        with open(attr_filename) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            voc = VocabularyTokens('train')
            
            for images in image_list:
                if images['split'] == 'train':
                    self.train_imgs.append(images['filename'])
                    self.train_imgs_id.append(images['imgid'])
                    
                    sentences = images['sentences']
                    sents_ids = []
                    
                    self.train_sentences[images['imgid']] = []
                    for sent in sentences:
                        self.train_sentences[images['imgid']].append(sent)
                        voc.add_sentence(sent['tokens'])
                    
                        
                elif images['split'] == 'val':
                    self.val_imgs.append(images['filename'])
                    self.val_imgs_id.append(images['imgid'])
                    
                    sentences = images['sentences']
                    sents_ids = []
                    
                    self.val_sentences[images['imgid']] = []
                    for sent in sentences:
                        self.val_sentences[images['imgid']].append(sent)
                        
                elif images['split'] == 'test':
                    self.test_imgs.append(images['filename'])
                    self.test_imgs_id.append(images['imgid'])
                    
                    sentences = images['sentences']
                    sents_ids = []
                    
                    self.test_sentences[images['imgid']] = []
                    for sent in sentences:
                        self.test_sentences[images['imgid']].append(sent)
                
            f.close()
            
            #voc.obtain_topK(self.K)
            voc.obtain_voc(self.T)            
            
            self.vocabulary = voc
            self.K = self.vocabulary.num_words
            
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(self.device)
            
            self.ntest = len(self.test_imgs_id)
            
    
    def next_batch(self, batch_size, attr = 'bert'):
        idx = torch.randperm(len(self.train_imgs_id))[0:batch_size]
        
        if attr == 'attributes':
            stop_words = set(nltk.corpus.stopwords.words('english')) 
            lemmatizer = WordNetLemmatizer() 
        
        #Obrir imatges
        self.zero_attrs = []
        batch_images = []
        batch_att = []
        batch_captions = []
        
        if attr == 'bert':
            self.bert.eval()
        
        j = 0
        for i in idx:
            imfile = os.path.join(self.image_path,self.train_imgs[i])
            #print(imfile)
            image = Image.open(imfile).resize((self.imagesize, self.imagesize))
            #image = torch.from_numpy(np.array(image, np.float32)).float()
            image = np.array(image, np.float32)[np.newaxis,...]
                
            
            sentences = self.train_sentences[self.train_imgs_id[i]]
            
            sent_idxs = []
            raws_r = []
            sentence_embedding_ar = []
            
            if attr == 'attributes':
                attr_vec = torch.zeros((1,self.K))
            elif attr == 'bert':
                attr_vec = torch.zeros((1, 768))
            
            sentix = np.random.randint(5)
            
            sent = sentences[sentix]
            raws = sent["raw"]
                
            if attr == 'attributes':
                sent_idxs = []
                    
                tokens = nltk.tokenize.word_tokenize(str(raws).lower())
                tokens_r = [w for w in tokens if not w in stop_words] 
                tokens_l = [lemmatizer.lemmatize(w) for w in tokens_r]
                    
                for token in tokens_l:
                    idxs = self.vocabulary.to_index(token)
                    if idxs != -1:
                        sent_idxs.append(idxs)
                    
                    attr_vec[:,sent_idxs] = 1
                    
                if all(attr_vec[0,:] == 0):
                    print('Attribute vector with all zeros')
                    continue
                
            elif attr == 'bert':
                input_ids = self.tokenizer.encode(str(raws).lower(), add_special_tokens = True)
                segments_ids = [1] * len(input_ids)
                
                tokens_tensor = torch.tensor([input_ids]).to(self.device)
                segments_tensors = torch.tensor([segments_ids]).to(self.device)
                
                with torch.no_grad():
                    outputs = self.bert(tokens_tensor, segments_tensors)
                    
                hidden_states = outputs[2]
                token_vecs = hidden_states[-2][0]
                sentence_embedding = torch.mean(token_vecs, dim=0)
                attr_vec[0,:] = sentence_embedding
                    
            
            if j == 0:
                batch_att = attr_vec
                batch_images = image
                #batch_captions = np.asarray(raws_r)
                #batch_captions = batch_captions[np.newaxis,...]
            else:
                #batch_captions = np.asarray(batch_captions)
                #raws_r = np.asarray(raws_r)
                #print(attr_vec.shape)
                batch_att = np.concatenate((batch_att,attr_vec), axis = 0)
                batch_images = np.asarray(batch_images)
                batch_images = np.concatenate((batch_images, image), axis = 0)
                #print(batch_att.shape)
                #print(raws_r)
                #batch_captions = np.concatenate((batch_captions,raws_r[np.newaxis,...]), axis = 0)   
                #print(batch_captions.shape)
            
            j = j + 1
        #batch_feature = self.data['train_seen']['resnet_features'][idx]   Canviat per les imatges entrades a la ResNet
        #batch_label =  self.data['train_seen']['labels'][idx]
        
        batch_images = torch.from_numpy(batch_images)
        batch_images = batch_images.permute(0,3,1,2)
        batch_att = torch.from_numpy(batch_att)
        
        print(batch_images.size())
        print(batch_att.size())
        
        
        return batch_images.to(self.device), batch_att.to(self.device), idx.to(self.device)
    
    def next_batch_test(self, batch_size, y, attr = 'bert'):
        idx = torch.arange(y*batch_size, (y + 1)*batch_size, 1)
        
        if attr == 'attributes':
            stop_words = set(nltk.corpus.stopwords.words('english')) 
            lemmatizer = WordNetLemmatizer() 
        
        batch_images = []
        batch_att = []
        batch_captions = []
        
        if attr == 'bert':
            self.bert.eval()
        
        j = 0
        for i in idx:
            imfile = os.path.join(self.image_path,self.test_imgs[i.item()])
            #print(imfile)
            image = Image.open(imfile).resize((self.imagesize, self.imagesize))
            #image = torch.from_numpy(np.array(image, np.float32)).float()
            image = np.array(image, np.float32)[np.newaxis,...]
            
            if j == 0:
                batch_images = image
            else:
                batch_images = np.asarray(batch_images)
                batch_images = np.concatenate((batch_images, image), axis = 0)
            
            sentences = self.test_sentences[self.test_imgs_id[i]]
            
            
            raws_r = []
            sentence_embedding_ar = []
            
            if attr == 'attributes':
                attr_vec = torch.zeros((5,self.K))
            elif attr == 'bert':
                attr_vec = torch.zeros((5, 768))
            
            k =0
            for sent in sentences:
                sent_idxs = []
                raws = sent["raw"]
                if attr == 'attributes':
                    tokens = nltk.tokenize.word_tokenize(str(raws).lower())
                    tokens_r = [w for w in tokens if not w in stop_words] 
                    tokens_l = [lemmatizer.lemmatize(w) for w in tokens_r]
                    
                    for token in tokens_l:
                        idxs = self.vocabulary.to_index(token)
                        if idxs != -1:
                            sent_idxs.append(idxs)
                    
                    attr_vec[k,sent_idxs] = 1
                    
                    if all(attr_vec[k,:] == 0):
                        print('Attribute vector with all zeros')
                
                elif attr == 'bert':
                    input_ids = self.tokenizer.encode(str(raws).lower(), add_special_tokens = True)
                    segments_ids = [1] * len(input_ids)
                
                    tokens_tensor = torch.tensor([input_ids]).to(self.device)
                    segments_tensors = torch.tensor([segments_ids]).to(self.device)
                
                    with torch.no_grad():
                        outputs = self.bert(tokens_tensor, segments_tensors)
                    
                    hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0]
                    sentence_embedding = torch.mean(token_vecs, dim=0)
                    attr_vec[k,:] = sentence_embedding
                    
                k = k + 1
                    
            
                       
            
            
            if j == 0:
                batch_att = attr_vec
                #batch_captions = np.asarray(raws_r)
                #batch_captions = batch_captions[np.newaxis,...]
            else:
                #batch_captions = np.asarray(batch_captions)
                #raws_r = np.asarray(raws_r)
                #print(attr_vec.shape)
                batch_att = np.concatenate((batch_att,attr_vec), axis = 0)
                #print(batch_att.shape)
                #print(raws_r)
                #batch_captions = np.concatenate((batch_captions,raws_r[np.newaxis,...]), axis = 0)   
                #print(batch_captions.shape)
            
            j = j + 1
        #batch_feature = self.data['train_seen']['resnet_features'][idx]   Canviat per les imatges entrades a la ResNet
        #batch_label =  self.data['train_seen']['labels'][idx]
        
        batch_images = torch.from_numpy(batch_images)
        
        batch_images = batch_images.permute(0,3,1,2)
        batch_att = torch.from_numpy(batch_att)
        
        
        return batch_images.to(self.device), batch_att.to(self.device), idx.to(self.device)
    
    def getZero(self):
        return len(self.zero_attrs)
            
    def get_item(self, idx, attr = 'bert'):
        if attr == 'attributes':
            stop_words = set(nltk.corpus.stopwords.words('english')) 
            lemmatizer = WordNetLemmatizer() 
        
        imfile = os.path.join(self.image_path,self.test_imgs[idx])
        
        image = Image.open(imfile).resize((self.imagesize, self.imagesize))
        image = np.array(image, np.float32)[np.newaxis,...]
        
        sentences = self.test_sentences[self.test_imgs_id[idx]]
        
        if attr == 'bert':
            self.bert.eval()
        
        sent_idxs = []
        raws_r = []
        sentence_embedding_ar = []
        
        if attr == 'attributes':
            attr_vec = torch.zeros((5,self.K))
        elif attr == 'bert':
            attr_vec = torch.zeros((5, 768))
            
        i =0
            
        for sent in sentences:
            raws = sent["raw"]
            if attr == 'attributes':
                tokens = nltk.tokenize.word_tokenize(str(raws).lower())
                tokens_r = [w for w in tokens if not w in stop_words] 
                tokens_l = [lemmatizer.lemmatize(w) for w in tokens_r]
                    
                for token in tokens_l:
                    idxs = self.vocabulary.to_index(token)
                    if idxs != -1:
                        sent_idxs.append(idxs)
                attr_vec[i,sent_idxs] = 1
                
                if all(attr_vec[i,:] == 0):
                    print('Attribute vector with all zeros')
                
            elif attr == 'bert':
                input_ids = self.tokenizer.encode(str(raws).lower(), add_special_tokens = True)
                segments_ids = [1] * len(input_ids)
                
                tokens_tensor = torch.tensor([input_ids]).to(self.device)
                segments_tensors = torch.tensor([segments_ids]).to(self.device)
                
                with torch.no_grad():
                    outputs = self.bert(tokens_tensor, segments_tensors)
                    
                hidden_states = outputs[2]
                token_vecs = hidden_states[-2][0]
                sentence_embedding = torch.mean(token_vecs, dim=0)
                attr_vec[i,:] = sentence_embedding
                    
            i = i + 1
            
        
                
        image = torch.from_numpy(image).permute(0,3,1,2)            
        
        return image.to(self.device), attr_vec.to(self.device), idx
            
        
    def __len__(self, typ = 'train'):
        return len(self.train_imgs_id)
        
class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda'):

        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        print('Project Directory:')
        print(project_directory)
        data_path = str(project_directory) + '/data'
        print('Data Path')
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        if self.dataset == 'CUB':
            self.datadir = self.data_path + '/CUB/'
        elif self.dataset == 'SUN':
            self.datadir = self.data_path + '/SUN/'
        elif self.dataset == 'AWA1':
            self.datadir = self.data_path + '/AWA1/'
        elif self.dataset == 'AWA2':
            self.datadir = self.data_path + '/AWA2/'


        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0


    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label =  self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [ batch_feature, batch_att]


    def read_matdataset(self):

        path= self.datadir + 'res101.mat'
        print('_____')
        print(path)
        matcontent = sio.loadmat(path)
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1

        path= self.datadir + 'att_splits.mat'
        matcontent = sio.loadmat(path)
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1


        if self.auxiliary_data_source == 'attributes':
            self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary datasource is not available for this dataset')
            else:

                with open(self.datadir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = torch.from_numpy(x[self.auxiliary_data_source]).float().to(self.device)


                print('loaded ', self.auxiliary_data_source)


        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.transform(feature[test_seen_loc])
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]
    
    def __len__(self):
        return self.ntrain

    def transfer_features(self, n, num_queries='num_features'):
        print('size before')
        print(self.data['test_unseen']['resnet_features'].size())
        print(self.data['train_seen']['resnet_features'].size())


        print('o'*100)
        print(self.data['test_unseen'].keys())
        for i,s in enumerate(self.novelclasses):

            features_of_that_class   = self.data['test_unseen']['resnet_features'][self.data['test_unseen']['labels']==s ,:]

            if 'attributes' == self.auxiliary_data_source:
                attributes_of_that_class = self.data['test_unseen']['attributes'][self.data['test_unseen']['labels']==s ,:]
                use_att = True
            else:
                use_att = False
            if 'sentences' == self.auxiliary_data_source:
                sentences_of_that_class = self.data['test_unseen']['sentences'][self.data['test_unseen']['labels']==s ,:]
                use_stc = True
            else:
                use_stc = False
            if 'word2vec' == self.auxiliary_data_source:
                word2vec_of_that_class = self.data['test_unseen']['word2vec'][self.data['test_unseen']['labels']==s ,:]
                use_w2v = True
            else:
                use_w2v = False
            if 'glove' == self.auxiliary_data_source:
                glove_of_that_class = self.data['test_unseen']['glove'][self.data['test_unseen']['labels']==s ,:]
                use_glo = True
            else:
                use_glo = False
            if 'wordnet' == self.auxiliary_data_source:
                wordnet_of_that_class = self.data['test_unseen']['wordnet'][self.data['test_unseen']['labels']==s ,:]
                use_hie = True
            else:
                use_hie = False


            num_features = features_of_that_class.size(0)

            indices = torch.randperm(num_features)

            if num_queries!='num_features':

                indices = indices[:n+num_queries]


            print(features_of_that_class.size())


            if i==0:

                new_train_unseen      = features_of_that_class[   indices[:n] ,:]

                if use_att:
                    new_train_unseen_att  = attributes_of_that_class[ indices[:n] ,:]
                if use_stc:
                    new_train_unseen_stc  = sentences_of_that_class[ indices[:n] ,:]
                if use_w2v:
                    new_train_unseen_w2v  = word2vec_of_that_class[ indices[:n] ,:]
                if use_glo:
                    new_train_unseen_glo  = glove_of_that_class[ indices[:n] ,:]
                if use_hie:
                    new_train_unseen_hie  = wordnet_of_that_class[ indices[:n] ,:]


                new_train_unseen_label  = s.repeat(n)

                new_test_unseen = features_of_that_class[  indices[n:] ,:]

                new_test_unseen_label = s.repeat( len(indices[n:] ))

            else:
                new_train_unseen  = torch.cat(( new_train_unseen             , features_of_that_class[  indices[:n] ,:]),dim=0)
                new_train_unseen_label  = torch.cat(( new_train_unseen_label , s.repeat(n)),dim=0)

                new_test_unseen =  torch.cat(( new_test_unseen,    features_of_that_class[  indices[n:] ,:]),dim=0)
                new_test_unseen_label = torch.cat(( new_test_unseen_label  ,s.repeat( len(indices[n:]) )) ,dim=0)

                if use_att:
                    new_train_unseen_att    = torch.cat(( new_train_unseen_att   , attributes_of_that_class[indices[:n] ,:]),dim=0)
                if use_stc:
                    new_train_unseen_stc    = torch.cat(( new_train_unseen_stc   , sentences_of_that_class[indices[:n] ,:]),dim=0)
                if use_w2v:
                    new_train_unseen_w2v    = torch.cat(( new_train_unseen_w2v   , word2vec_of_that_class[indices[:n] ,:]),dim=0)
                if use_glo:
                    new_train_unseen_glo    = torch.cat(( new_train_unseen_glo   , glove_of_that_class[indices[:n] ,:]),dim=0)
                if use_hie:
                    new_train_unseen_hie    = torch.cat(( new_train_unseen_hie   , wordnet_of_that_class[indices[:n] ,:]),dim=0)



        print('new_test_unseen.size(): ', new_test_unseen.size())
        print('new_test_unseen_label.size(): ', new_test_unseen_label.size())
        print('new_train_unseen.size(): ', new_train_unseen.size())
        #print('new_train_unseen_att.size(): ', new_train_unseen_att.size())
        print('new_train_unseen_label.size(): ', new_train_unseen_label.size())
        print('>> num novel classes: ' + str(len(self.novelclasses)))

        #######
        ##
        #######

        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)
        #self.data['train_seen']['resnet_features']  = copy.deepcopy(new_train_seen)

        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)
        #self.data['train_seen']['labels']  = copy.deepcopy(new_train_seen_label)

        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data['train_unseen']['resnet_features'].size(0)

        if use_att:
            self.data['train_unseen']['attributes'] = copy.deepcopy(new_train_unseen_att)
        if use_w2v:
            self.data['train_unseen']['word2vec']   = copy.deepcopy(new_train_unseen_w2v)
        if use_stc:
            self.data['train_unseen']['sentences']  = copy.deepcopy(new_train_unseen_stc)
        if use_glo:
            self.data['train_unseen']['glove']      = copy.deepcopy(new_train_unseen_glo)
        if use_hie:
            self.data['train_unseen']['wordnet']   = copy.deepcopy(new_train_unseen_hie)

        ####
        self.data['train_seen_unseen_mixed'] = {}
        self.data['train_seen_unseen_mixed']['resnet_features'] = torch.cat((self.data['train_seen']['resnet_features'],self.data['train_unseen']['resnet_features']),dim=0)
        self.data['train_seen_unseen_mixed']['labels'] = torch.cat((self.data['train_seen']['labels'],self.data['train_unseen']['labels']),dim=0)

        self.ntrain_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)

        if use_att:
            self.data['train_seen_unseen_mixed']['attributes'] = torch.cat((self.data['train_seen']['attributes'],self.data['train_unseen']['attributes']),dim=0)
        if use_w2v:
            self.data['train_seen_unseen_mixed']['word2vec'] = torch.cat((self.data['train_seen']['word2vec'],self.data['train_unseen']['word2vec']),dim=0)
        if use_stc:
            self.data['train_seen_unseen_mixed']['sentences'] = torch.cat((self.data['train_seen']['sentences'],self.data['train_unseen']['sentences']),dim=0)
        if use_glo:
            self.data['train_seen_unseen_mixed']['glove'] = torch.cat((self.data['train_seen']['glove'],self.data['train_unseen']['glove']),dim=0)
        if use_hie:
            self.data['train_seen_unseen_mixed']['wordnet'] = torch.cat((self.data['train_seen']['wordnet'],self.data['train_unseen']['wordnet']),dim=0)

#d = DATA_LOADER()
