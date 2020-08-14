#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:27:35 2020

@author: sergi
"""

import os
from pathlib import Path
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
from torchvision import models as torchModels
from torchvision import transforms
from PIL import Image
from vocabulary import VocabularyTokens
from transformers import BertTokenizer, BertModel

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

dataset = 'MSCOCO' #'MSCOCO', 'VizWiz'

device = 'cuda'
imagesize = 224

print("The current working directory is")
print(os.getcwd())
folder = str(Path(os.getcwd()))
if folder[-5:] == 'model':
    project_directory = Path(os.getcwd()).parent
else:
    project_directory = folder
        
path = os.path.join(project_directory,'data',dataset)
        
image_path = path
print('The images folder is')
print(image_path)
        
device = device
index_in_epoch = 0
epochs_completed = 0
        
#self.K = 50
T = 150
        
attr = 'bert'
        
transforms = transforms.Compose([ 
        transforms.Resize(imagesize),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
#feature_extractor_all = torchModels.resnet101(pretrained = True)
#feature_extractor = nn.Sequential(*list(feature_extractor_all.children())[:-2])

feature_extractor = torchModels.resnet101(pretrained = True)
        
num_ftrs = feature_extractor.fc.in_features
feature_extractor.fc = Identity()
feature_extractor.to(device)
        
train_imgs = []
train_imgs_id = []
train_sentences = {}
train_sents_ids = {}
        
val_imgs = []
val_imgs_id = []
val_sentences = {}
val_sents_ids = {}
        
test_imgs = []
test_imgs_id = []
test_sentences = {}
test_sents_ids = {}
        
print(path)
attr_filename = os.path.join(path,'annotations', 'captions_train2014.json')

print('Loading train')
with open(attr_filename) as f:
    json_data = json.loads(f.read())
    image_list = json_data['images']
    captions_list = json_data['annotations']
            
    voc = VocabularyTokens('train')
            
    for images in image_list:
        train_imgs.append(images['file_name'])
        train_imgs_id.append(images['id'])
        
        print(str(images['id']))
        train_sentences[images['id']] = []
        for caption in captions_list:
            if caption['image_id'] == images['id']:
                train_sentences[images['id']].append(caption)
                
                if attr == 'attributes':
                    sentence = caption['caption']
                    tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
                    tokens_r = [w for w in tokens if not w in stop_words] 
                    tokens_l = [lemmatizer.lemmatize(w) for w in tokens_r]
                
                    voc.add_sentence(tokens_l)
                    
    f.close()

print('Loading validation/test')
attr_filename = os.path.join(path,'annotations', 'captions_val2014.json')
with open(attr_filename) as f:
    json_data = json.loads(f.read())
    image_list = json_data['images']
    captions_list = json_data['annotations']
            
    voc = VocabularyTokens('train')
            
    for images in image_list:
        val_imgs.append(images['file_name'])
        val_imgs_id.append(images['id'])
        
        print(str(images['id']))
        
        val_sentences[images['id']] = []
        for caption in captions_list:
            if caption['image_id'] == images['id']:
                val_sentences[images['id']].append(caption)
                    
    f.close()

'''
attr_filename = os.path.join(path,'annotations', 'image_info_test2014.json')
with open(attr_filename) as f:
    json_data = json.loads(f.read())
    image_list = json_data['images']
    captions_list = json_data['annotations']
            
    voc = VocabularyTokens('train')
            
    for images in image_list:
        test_imgs.append(images['file_name'])
        test_imgs_id.append(images['id'])
        
        test_sentences[images['id']] = []
        for caption in captions_list:
            if caption['image_id'] == images['id']:
                test_sentences[images['id']].append(caption)
                
                    
    f.close()
'''   
    # '''                
    # elif images['split'] == 'val':
    #     val_imgs.append(images['filename'])
    #     val_imgs_id.append(images['imgid'])
                    
    #     sentences = images['sentences']
    #     sents_ids = []
                    
    #     val_sentences[images['imgid']] = []
    #     for sent in sentences:
    #         val_sentences[images['imgid']].append(sent)
                        
    # elif images['split'] == 'test':
    #     test_imgs.append(images['filename'])
    #     test_imgs_id.append(images['imgid'])
                    
    #     sentences = images['sentences']
    #     sents_ids = []
                    
    #     test_sentences[images['imgid']] = []
    #     for sent in sentences:
    #         test_sentences[images['imgid']].append(sent)'''
                
    
            
#voc.obtain_topK(self.K)
voc.obtain_voc(T)            
            
vocabulary = voc
K = vocabulary.num_words
            
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device)
            
ntest = len(test_imgs_id)

print('Generating features for training')
y = 0
for i in range(0, len(train_imgs_id), 20):
    
    print(i)
        
    #Obrir imatges
    zero_attrs = []
    batch_images = []
    batch_att = []
    batch_captions = []
        
    if attr == 'bert':
        bert.eval()
    
    idx = np.arange(i,i + 20)
    
    j = 0
    for x in idx:
        
        if x >= len(train_imgs):
            continue
        
        imfile = os.path.join(image_path,'train2014',train_imgs[x])
        #print(imfile)
        image = Image.open(imfile).resize((imagesize, imagesize)).convert('RGB')
        #image = torch.from_numpy(np.array(image, np.float32)).float()
        image = np.array(image, np.float32)[np.newaxis,...]
                
            
        sentences = train_sentences[train_imgs_id[x]]
            
        sent_idxs = []
        raws_r = []
        sentence_embedding_ar = []
            
        if attr == 'attributes':
            attr_vec = torch.zeros((5,K))
        elif attr == 'bert':
            attr_vec = torch.zeros((5, 768))
            
        k = 0
        
        print(len(sentences))
        for sent in sentences:
            raws = sent["caption"]
            
            if k >= 5:
                continue
            
            if attr == 'attributes':
                sent_idxs = []
                    
                tokens = nltk.tokenize.word_tokenize(str(raws).lower())
                tokens_r = [w for w in tokens if not w in stop_words] 
                tokens_l = [lemmatizer.lemmatize(w) for w in tokens_r]
                    
                for token in tokens_l:
                    idxs = vocabulary.to_index(token)
                    if idxs != -1:
                        sent_idxs.append(idxs)
                    
                    attr_vec[k,sent_idxs] = 1
                    
                if all(attr_vec[k,:] == 0):
                    print('Attribute vector with all zeros')
                
            elif attr == 'bert':
                input_ids = tokenizer.encode(str(raws).lower(), add_special_tokens = True)
                segments_ids = [1] * len(input_ids)
                
                tokens_tensor = torch.tensor([input_ids]).to(device)
                segments_tensors = torch.tensor([segments_ids]).to(device)
                
                with torch.no_grad():
                    outputs = bert(tokens_tensor, segments_tensors)
                    
                hidden_states = outputs[2]
                token_vecs = hidden_states[-2][0]
                sentence_embedding = torch.mean(token_vecs, dim=0)
                
                print(k)
                attr_vec[k,:] = sentence_embedding
            
            k += 1
                    
            
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
            print(batch_images.shape)
            print(image.shape)
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
    
    with torch.no_grad():
        features = feature_extractor(batch_images.to(device)) 
    
    batch_att = torch.from_numpy(batch_att)
    
    if y == 0:
        total_features = features.cpu()
        total_att = batch_att
    else:
        total_features = torch.cat((total_features, features.cpu()), dim = 0)
        total_att = torch.cat((total_att, batch_att), dim = 0)
    
    y = y + 1


print('Saving features for training')
if attr == 'attributes':
    with open('attr_training_mscoco.pkl','wb') as ft:
        pickle.dump([total_att], ft)
        ft.close()
    
    with open('ft_attributes_training_mscoco.pkl','wb') as f:
        pickle.dump([total_features], f)
        f.close()
        
elif attr == 'bert':
    with open('ft_bert_training_mscoco.pkl','wb') as f:
        pickle.dump([total_features], f)
        f.close()
        
    with open('bert_training_mscoco.pkl','wb') as ft:
        pickle.dump([total_att], ft)
        ft.close()

print('Generating features for test')
y = 0
for i in range(0, ntest, 20):
    if attr == 'attributes':
        stop_words = set(nltk.corpus.stopwords.words('english')) 
        lemmatizer = WordNetLemmatizer() 
    
    print(i)
    
    #Obrir imatges
    zero_attrs = []
    batch_images = []
    batch_att = []
    batch_captions = []
        
    if attr == 'bert':
        bert.eval()
    
    idx = np.arange(i,i + 20)
    
    j = 0
    for x in idx:
        
        if x >= len(train_imgs):
            continue
        
        imfile = os.path.join(image_path,'test2014',test_imgs[x])
        #print(imfile)
        image = Image.open(imfile).resize((imagesize, imagesize)).convert('RGB')
        #image = torch.from_numpy(np.array(image, np.float32)).float()
        image = np.array(image, np.float32)[np.newaxis,...]
                
            
        sentences = test_sentences[test_imgs_id[x]]
            
        sent_idxs = []
        raws_r = []
        sentence_embedding_ar = []
            
        if attr == 'attributes':
            attr_vec = torch.zeros((5,K))
        elif attr == 'bert':
            attr_vec = torch.zeros((5, 768))
            
        print(len(sentences))
        k = 0
        for sent in sentences:
            raws = sent["caption"]
            
            if k >= 5:
                continue
            
            if attr == 'attributes':
                sent_idxs = []
                    
                tokens = nltk.tokenize.word_tokenize(str(raws).lower())
                tokens_r = [w for w in tokens if not w in stop_words] 
                tokens_l = [lemmatizer.lemmatize(w) for w in tokens_r]
                    
                for token in tokens_l:
                    idxs = vocabulary.to_index(token)
                    if idxs != -1:
                        sent_idxs.append(idxs)
                    
                attr_vec[k,sent_idxs] = 1
                
            elif attr == 'bert':
                input_ids = tokenizer.encode(str(raws).lower(), add_special_tokens = True)
                segments_ids = [1] * len(input_ids)
                
                tokens_tensor = torch.tensor([input_ids]).to(device)
                segments_tensors = torch.tensor([segments_ids]).to(device)
                
                with torch.no_grad():
                    outputs = bert(tokens_tensor, segments_tensors)
            
                hidden_states = outputs[2]
                token_vecs = hidden_states[-2][0]
                sentence_embedding = torch.mean(token_vecs, dim=0)
                print(k)
                attr_vec[k,:] = sentence_embedding
            
            k += 1
            
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
    
    with torch.no_grad():
        features = feature_extractor(batch_images.to(device)) 
    
    batch_att = torch.from_numpy(batch_att)
    
    if y == 0:
        total_features = features.cpu()
        total_att = batch_att
    else:
        total_features = torch.cat((total_features, features.cpu()), dim = 0)
        total_att = torch.cat((total_att, batch_att), dim = 0)
    
    y = y + 1

print('Saving features for test')
if attr == 'attributes':
    with open('attr_test_mscoco.pkl','wb') as ft:
        pickle.dump([total_att], ft)
        ft.close()
    
    with open('ft_attributes_test_mscoco.pkl','wb') as f:
        pickle.dump([total_features], f)
        f.close()
        
elif attr == 'bert':
    with open('ft_bert_test_mscoco.pkl','wb') as f:
        pickle.dump([total_features], f)
        f.close()
        
    with open('bert_test_mscoco.pkl','wb') as ft:
        pickle.dump([total_att], ft)
        ft.close()
    

