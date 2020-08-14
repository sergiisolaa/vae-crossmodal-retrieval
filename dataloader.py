#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:45:16 2020

@author: sergi
"""

import os
from pathlib import Path
import pickle
import numpy as np
import torch
from torchvision import transforms


class Flickr30k(object):
    def __init__(self, path, device = 'cuda', imagesize = 224, attr = 'attributes'):
        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        
        self.path = os.path.join(folder,attr,'flickr30k')
        
        self.device = device
        
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
        self.imagesize = imagesize
        
        self.transforms = transforms.Compose([ 
            transforms.Resize(self.imagesize),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
        self.attr = attr
        
        filename = os.path.join(self.path,'ft_training.pkl')
        with open(filename, 'rb') as f:
            self.fts_train = pickle.load(f)
        
        filename = os.path.join(self.path,'ft_test.pkl')
        with open(filename, 'rb') as f:
            self.fts_test = pickle.load(f)
        
        if self.attr == 'attributes':
            atfilename = os.path.join(self.path,'attr_training.pkl')
        elif self.attr == 'bert':
            atfilename = os.path.join(self.path,'bert_training.pkl')
        
        with open(atfilename, 'rb') as fat:
            self.attrs_train = pickle.load(fat)
        
        if self.attr == 'attributes':
            atfilename = os.path.join(self.path,'attr_test.pkl')
        elif self.attr == 'bert':
            atfilename = os.path.join(self.path,'bert_test.pkl')
        
        with open(atfilename, 'rb') as fat:
            self.attrs_test = pickle.load(fat)
        
        
        self.K = self.attrs_train[0].shape[1]
        
        self.ntest = len(self.fts_test[0])
    
    def __len__(self):
        return len(self.fts_train[0])
        
    def next_batch(self,batch_size):       
          
        idx = torch.randperm(len(self.fts_train[0]))[0:batch_size]
        j = 0
        for i in idx:
            img = self.fts_train[0][i,:]
            attrs = self.attrs_train[0][5*i:5*i + 5,:]            
            randi = torch.randperm(5)[0]
            att = attrs[randi,:].unsqueeze(0) 
            
            if len(torch.nonzero(att[0,:])) == 0:
                continue
            
            if j == 0:   
                batch_att = att
                batch_imgs = img.unsqueeze(0)
            else:
                batch_att = torch.cat((batch_att, att), dim = 0)
                batch_imgs = torch.cat((batch_imgs, img.unsqueeze(0)), dim = 0)
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def next_batch2(self,batch_size):       
          
        idx = torch.randperm(len(self.fts_test[0]))[0:batch_size]
        j = 0
        for i in idx:
            img = self.fts_test[0][i,:]
            attrs = self.attrs_test[0][5*i:5*i + 5,:]            
            randi = torch.randperm(5)[0]
            att = attrs[randi,:].unsqueeze(0) 
            
            if len(torch.nonzero(att[0,:])) == 0:
                continue
            
            if j == 0:   
                batch_att = att
                batch_imgs = img.unsqueeze(0)
            else:
                batch_att = torch.cat((batch_att, att), dim = 0)
                batch_imgs = torch.cat((batch_imgs, img.unsqueeze(0)), dim = 0)
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def next_batch_test(self,batch_size, y):       
          
        idx = torch.arange(y*batch_size, (y + 1)*batch_size, 1)
        
        batch_imgs = self.fts_test[0][idx,:]
        
        idx_at = 5*idx 
        j = 0
        for i in idx_at:
            attrs = self.attrs_test[0][i:i + 5,:]
            
            if j == 0:
                batch_att = attrs
            else:
                batch_att = torch.cat((batch_att, attrs), dim = 0)
            
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def get_item(self, idx):
        
        image = self.fts_test[0][idx,:]
        image = image.unsqueeze(0)
        
        ida = 5*idx
        attrs = self.attrs_test[0][ida:ida + 5,:]
        
        return image, attrs, idx

class MSCOCO(object):
    def __init__(self, path, device = 'cuda', imagesize = 224, attr = 'attributes'):
        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        
        self.path = os.path.join(folder,attr,'mscoco')
        
        self.device = device
        
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
        self.imagesize = imagesize
        
        self.transforms = transforms.Compose([ 
            transforms.Resize(self.imagesize),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
        self.attr = attr
        
        filename = os.path.join(self.path,'ft_training.pkl')
        with open(filename, 'rb') as f:
            self.fts_train = pickle.load(f)
        
        filename = os.path.join(self.path,'ft_test.pkl')
        with open(filename, 'rb') as f:
            self.fts_test = pickle.load(f)
        
        if self.attr == 'attributes':
            atfilename = os.path.join(self.path,'attr_training.pkl')
        elif self.attr == 'bert':
            atfilename = os.path.join(self.path,'bert_training.pkl')
        
        with open(atfilename, 'rb') as fat:
            self.attrs_train = pickle.load(fat)
        
        if self.attr == 'attributes':
            atfilename = os.path.join(self.path,'attr_test.pkl')
        elif self.attr == 'bert':
            atfilename = os.path.join(self.path,'bert_test.pkl')
        
        with open(atfilename, 'rb') as fat:
            self.attrs_test = pickle.load(fat)
        
        
        self.K = self.attrs_train[0].shape[1]
        
        self.ntest = len(self.fts_test[0])
    
    def __len__(self):
        return len(self.fts_train[0])
        
    def next_batch(self,batch_size):       
          
        idx = torch.randperm(len(self.fts_train[0]))[0:batch_size]
        j = 0
        for i in idx:
            img = self.fts_train[0][i,:]
            attrs = self.attrs_train[0][5*i:5*i + 5,:]            
            randi = torch.randperm(5)[0]
            att = attrs[randi,:].unsqueeze(0) 
            
            if len(torch.nonzero(att[0,:])) == 0:
                continue
            
            if j == 0:   
                batch_att = att
                batch_imgs = img.unsqueeze(0)
            else:
                batch_att = torch.cat((batch_att, att), dim = 0)
                batch_imgs = torch.cat((batch_imgs, img.unsqueeze(0)), dim = 0)
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def next_batch2(self,batch_size):       
          
        idx = torch.randperm(len(self.fts_test[0]))[0:batch_size]
        j = 0
        for i in idx:
            img = self.fts_test[0][i,:]
            attrs = self.attrs_test[0][5*i:5*i + 5,:]            
            randi = torch.randperm(5)[0]
            att = attrs[randi,:].unsqueeze(0) 
            
            if len(torch.nonzero(att[0,:])) == 0:
                continue
            
            if j == 0:   
                batch_att = att
                batch_imgs = img.unsqueeze(0)
            else:
                batch_att = torch.cat((batch_att, att), dim = 0)
                batch_imgs = torch.cat((batch_imgs, img.unsqueeze(0)), dim = 0)
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def next_batch_test(self,batch_size, y):       
          
        idx = torch.arange(y*batch_size, (y + 1)*batch_size, 1)
        
        batch_imgs = self.fts_test[0][idx,:]
        
        idx_at = 5*idx 
        j = 0
        for i in idx_at:
            attrs = self.attrs_test[0][i:i + 5,:]
            
            if j == 0:
                batch_att = attrs
            else:
                batch_att = torch.cat((batch_att, attrs), dim = 0)
            
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def get_item(self, idx):
        
        image = self.fts_test[0][idx,:]
        image = image.unsqueeze(0)
        
        ida = 5*idx
        attrs = self.attrs_test[0][ida:ida + 5,:]
        
        return image, attrs, idx
    
    
class VizWiz(object):
    def __init__(self, path, device = 'cuda', imagesize = 224, attr = 'attributes'):
        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        
        self.path = os.path.join(folder,attr,'vizwiz')
        
        self.device = device
        
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
        self.imagesize = imagesize
        
        self.transforms = transforms.Compose([ 
            transforms.Resize(self.imagesize),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
        self.attr = attr
        
        filename = os.path.join(self.path,'ft_training.pkl')
        with open(filename, 'rb') as f:
            self.fts_train = pickle.load(f)
        
        filename = os.path.join(self.path,'ft_test.pkl')
        with open(filename, 'rb') as f:
            self.fts_test = pickle.load(f)
        
        if self.attr == 'attributes':
            atfilename = os.path.join(self.path,'attr_training.pkl')
        elif self.attr == 'bert':
            atfilename = os.path.join(self.path,'bert_training.pkl')
        
        with open(atfilename, 'rb') as fat:
            self.attrs_train = pickle.load(fat)
        
        if self.attr == 'attributes':
            atfilename = os.path.join(self.path,'attr_test.pkl')
        elif self.attr == 'bert':
            atfilename = os.path.join(self.path,'bert_test.pkl')
        
        with open(atfilename, 'rb') as fat:
            self.attrs_test = pickle.load(fat)
        
        
        self.K = self.attrs_train[0].shape[1]
        
        self.ntest = len(self.fts_test[0])
    
    def __len__(self):
        return len(self.fts_train[0])
        
    def next_batch(self,batch_size):       
          
        idx = torch.randperm(len(self.fts_train[0]))[0:batch_size]
        j = 0
        for i in idx:
            img = self.fts_train[0][i,:]
            attrs = self.attrs_train[0][5*i:5*i + 5,:]            
            randi = torch.randperm(5)[0]
            att = attrs[randi,:].unsqueeze(0) 
            
            if len(torch.nonzero(att[0,:])) == 0:
                continue
            
            if j == 0:   
                batch_att = att
                batch_imgs = img.unsqueeze(0)
            else:
                batch_att = torch.cat((batch_att, att), dim = 0)
                batch_imgs = torch.cat((batch_imgs, img.unsqueeze(0)), dim = 0)
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def next_batch2(self,batch_size):       
          
        idx = torch.randperm(len(self.fts_test[0]))[0:batch_size]
        j = 0
        for i in idx:
            img = self.fts_test[0][i,:]
            attrs = self.attrs_test[0][5*i:5*i + 5,:]            
            randi = torch.randperm(5)[0]
            att = attrs[randi,:].unsqueeze(0) 
            
            if len(torch.nonzero(att[0,:])) == 0:
                continue
            
            if j == 0:   
                batch_att = att
                batch_imgs = img.unsqueeze(0)
            else:
                batch_att = torch.cat((batch_att, att), dim = 0)
                batch_imgs = torch.cat((batch_imgs, img.unsqueeze(0)), dim = 0)
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def next_batch_test(self,batch_size, y):       
          
        idx = torch.arange(y*batch_size, (y + 1)*batch_size, 1)
        
        batch_imgs = self.fts_test[0][idx,:]
        
        idx_at = 5*idx 
        j = 0
        for i in idx_at:
            attrs = self.attrs_test[0][i:i + 5,:]
            
            if j == 0:
                batch_att = attrs
            else:
                batch_att = torch.cat((batch_att, attrs), dim = 0)
            
            j+= 1
        
        return batch_imgs, batch_att, idx
    
    def get_item(self, idx):
        
        image = self.fts_test[0][idx,:]
        image = image.unsqueeze(0)
        
        ida = 5*idx
        attrs = self.attrs_test[0][ida:ida + 5,:]
        
        return image, attrs, idx
    
    
            
        
        