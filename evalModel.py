# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:24:06 2020

@author: PC
"""

import random
from pathlib import Path
import json
import os

import numpy as np
from PIL import Image
from scipy.spatial import distance

import torch
import torch.nn as nn
from torchvision import models as torchModels
from torchvision import transforms

from transformers import BertTokenizer, BertModel

from vaemodelTriplet import Model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Evaluate:   
    
    

    def selectEvalItems(self,path):
    
        img_id = random.randint(0,1000)
        sent_id = random.randint(0,5000)
        
        self.path = path
        self.filename = os.path.join(path,'dataset.json')
        
        with open(self.filename) as f:
            json_data = json.loads(f.read())
            self.bbdd = json_data['images']            
            
            image = self.bbdd[img_id]
            
            im_filename = image['filename']
            
            im = Image.open(os.path.join(path,'images',im_filename))
            
            fn = 'query' + str(img_id) + '.png'
            im.save(fn)
            
            image2 = self.bbdd[sent_id//5]
            mod = sent_id%5
            
            captions = image2['sentences']
            caption = captions[mod]['raw']
            
            print(caption)
            
            print('IMG ID: ', str(img_id))
            print('SENT ID: ', str(sent_id), ' from image ', str(sent_id//5))
        
        return im, img_id, caption, sent_id
        
    
    def evalI2T(self,model, im, img_id):
        
        self.device = 'cuda'
        
        model.eval()
        
        model.generate_gallery()
        
        feature_extractor = torchModels.resnet101(pretrained = True)
            
        num_ftrs = feature_extractor.fc.in_features
        feature_extractor.fc = Identity()
        feature_extractor.to(self.device)
        
        trans1 = transforms.ToTensor()
        im = trans1(im).unsqueeze(0).to(self.device)
        
        im_ft = feature_extractor(im)
                
        mu_img, logvar_img, z_from_img = model.obtain_embeds(im_ft, 'image')
        
        distances = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
        indices = np.argsort(distances)
        
        
        for i in range(0, 10):
            idx = indices[:,i]
            im_id = idx//5
            
            im = self.bbdd[int(im_id)]
            captions = im['sentences']
            caption = captions[int(idx%5)]['raw']
            
            print(caption)
        
    
    def evalT2I(self,model, caption, sent_id):
        
        model.eval()
        
        model.generate_gallery()            
        
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(self.device)
        
        bert.eval()
        
        input_ids = tokenizer.encode(str(caption).lower(), add_special_tokens = True)
        segments_ids = [1] * len(input_ids)
                
        tokens_tensor = torch.tensor([input_ids]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
                
        with torch.no_grad():
            outputs = bert(tokens_tensor, segments_tensors)
                    
        hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        caption = torch.mean(token_vecs, dim=0)
                
        mu_att, logvar_att, z_from_att = model.obtain_embeds(caption.unsqueeze(0), 'attributes')
        
        distances = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
        indices = np.argsort(distances)
        
        for i in range(0, 10):
            idx = indices[:,i]
            
            im_id = idx//5
            
            im = self.bbdd[int(im_id)]
            imgfn = im['filename']
            
            imgs = Image.open(os.path.join(self.path,'images',imgfn))
            
            fn = 'r'+str(i)+'-'+str(sent_id)+'.png'
            imgs = imgs.save(fn)
            
            print('Image plotted')
    
    def evalRetrieval(self, model, printTop = 1):
        
        model.eval()
        
        ranksI = np.zeros((1,5*model.dataset.ntest))
        ranksT = np.zeros((1,model.dataset.ntest))    
        
        for i in range(0, model.dataset.ntest):
            
            mu_img = model.gallery_imgs_z[i,:].unsqueeze(0)  
            mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
            
            distancesI = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
            distancesT = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
            
            indicesI = np.argsort(distancesI)
            indicesT = np.argsort(distancesT[0,:])
                       
                        
            for z in range(0,5):
                if len(indicesI[z] == i) != 0:
                    ranksI[:,(5*i) + z] = np.where(indicesI[z] == i)[0][0]
                else:
                    ranksI[:,(5*i) + z] = 1000
                
                if ranksI[:,(5*i) + z] < printTop:
                    
                    for j in range(0,10):
                        im = self.bbdd[indicesI[z][j]]['filename']
                    
                        img = Image.open(os.path.join(self.path, 'images', im))
                    
                        
                    
                        fn = 'r'+str(j)+'-'+str((5*i)+z)+'.png'
                        print(fn)
                        img.save(fn)
                    
                    captions = self.bbdd[int(i)]['sentences']
                    caption = captions[z]
                    
                    print('QUERY', str(((5*i)+z)))
                    print(caption['raw'])
            
            
            if len(np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))) != 0:
                ranksT[:,i] = np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))[0][0]
            else:
                ranksT[:,i] = 1000
                
            if ranksT[:,i] < printTop:
                im = self.bbdd[int(i)]['filename']
                
                img = Image.open(os.path.join(self.path, 'images', im))               
                
                
                fn = 'query'+str(i)+'.png'
                print(fn)
                img.save(fn)
                
                for j in range(0,10):
                    captions = self.bbdd[indicesT[j] // 5]['sentences']
                    caption = captions[indicesT[j]%5]
                    
                    print(caption['raw'])
        
        #Fer el retrieval de tots per tots
        #Si printTop fer plot, save i print dels top1 (query i retrieved)
        
        
        
        
        
        
 
