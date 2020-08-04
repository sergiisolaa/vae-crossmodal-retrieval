# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:24:06 2020

@author: PC
"""

import random
from pathlib import Path
import json
import os

from PIL import Image
from scipy.spatial import distance

import torch
import torch.nn as nn
from torchvision import models as torchModels

from vaemodelTriplet import Model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Evaluate:   

    def selectEvalItems(path):
    
        img_id = random.randint(0,1000)
        sent_id = random.randint(0,5000)
        
        self.path = path
        self.filename = os.path.join(path,'dataset.json')
        
        with open(filename) as f:
            json_data = json.loads(f.read())
            self.bbdd = json_data['images']            
            
            image = bbdd[img_id]
            
            im_filename = image['filename']
            
            im = Image.open(os.path.join(path,'images',im_filename))
            im.show()
            
            image2 = bbdd[sent_id//5]
            mod = sent_id%5
            
            captions = image2['sentences']
            caption = captions[mod]['raw']
            
            print(caption)
        
        return im, img_id, caption, sent_id
        
    
    def evalI2T(model, im, img_id):
        
        model.eval()
        
        model.generate_gallery()
        
        feature_extractor = torchModels.resnet101(pretrained = True)
            
        num_ftrs = feature_extractor.fc.in_features
        feature_extractor.fc = Identity()
        feature_extractor.to(device)
        
        im_ft = feature_extractor(im)
                
        mu_img, logvar_img, z_from_img = obtain_embeds(im_ft, 'image')
        
        distances = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
        indices = np.argsort(distances)
        
        for i in range(0, 10):
            idx = indices[i]
            
            im_id = idx//5
            
            im = self.bbdd[im_id]
            captions = im['sentences']
            caption = captions[idx%5]['raw']
            
            print(caption)
        
    
    def evalT2I(model, caption, sent_id):
        
        model.eval()
        
        model.generate_gallery()            
        
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device)
        
        bert.eval()
        
        input_ids = tokenizer.encode(str(raws).lower(), add_special_tokens = True)
        segments_ids = [1] * len(input_ids)
                
        tokens_tensor = torch.tensor([input_ids]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
                
        with torch.no_grad():
            outputs = bert(tokens_tensor, segments_tensors)
                    
        hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        caption = torch.mean(token_vecs, dim=0)
                
        mu_att, logvar_att, z_from_att = obtain_embeds(caption, 'attributes')
        
        distances = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
        indices = np.argsort(distances)
        
        for i in range(0, 10):
            idx = indices[i]
            
            im_id = idx//5
            
            im = self.bbdd[im_id]
            imgfn = im['filename']
            
            imgs = Image.open(os.path.join(path,'images',imgfn))
            im.show()
            
            print(Image plotted)
            
        
        
        
 
