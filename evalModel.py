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
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torchvision import models as torchModels
from torchvision import transforms

from transformers import BertTokenizer, BertModel

from vaemodelTriplet import Model

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from sklearn.preprocessing import normalize

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
            
            #print(caption)
        
    
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
            
            #print('Image plotted')
    
    def tSNEretrieval(self, model, printTop = 1):
        model.eval()
        
        test_imgs = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    test_imgs.append(images['filename'])
                    
                
        
        print('here i am')
        
        ranksI = np.zeros((1,model.dataset.ntest))
        ranksT = np.zeros((1,5*model.dataset.ntest))    
        
        txt = 'Retrieval-'+str(printTop)+'.txt'
        
        train_samples = np.vstack((model.gallery_imgs_z.clone().cpu().detach(),model.gallery_attrs_z.clone().cpu().detach()))
        tsne = TSNE(n_components = 2)
        
        embedded = tsne.fit_transform(train_samples)
        
        z_imgs_embedded = embedded[0:1000,:]
        z_attrs_embedded = embedded[1000:6000,:]
        
        tx_imgs = z_imgs_embedded[:,0]
        ty_imgs = z_imgs_embedded[:,1]
        tx_attrs = z_attrs_embedded[:,0]
        ty_attrs = z_attrs_embedded[:,1]
        
        minx = min(np.min(tx_imgs), np.min(tx_attrs))
        maxx = max(np.max(tx_imgs), np.max(tx_attrs))
        miny = min(np.min(ty_imgs), np.min(ty_attrs))
        maxy = max(np.max(ty_imgs), np.max(ty_attrs))
        
        tx_imgs = (tx_imgs-minx)/(maxx-minx)
        ty_imgs = (ty_imgs-miny)/(maxy-miny)
        tx_attrs =(tx_attrs-minx)/(maxx-minx)
        ty_attrs =(ty_attrs-miny)/(maxy-miny)
        
               
        with open(txt, 'w') as txtfile:
            
            indixI = []
            indixT = []
            
            ins = []
            for i in range(0, model.dataset.ntest):
                
                mu_img = model.gallery_imgs_z[i,:].unsqueeze(0)  
                mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
                
                #distancesI = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy())
                distancesI = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy())
                #distancesT = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy())
                distancesT = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy())
                
                print(distancesI[0,:])
                
                indicesI = np.argsort(distancesI)
                indicesT = np.argsort(distancesT)
                
                print(indicesI[0,:])
                
                print('Ordered distances')
                
                print(distancesI[0,indicesI[0,:]])
                
                
                for j in range(0,indicesI.shape[0]):
                    full_image = Image.new('RGBA', (4000, 3000))
                    
                    for k in range(0,20):
                        tile = Image.open(os.path.join(self.path,'images', test_imgs[indicesI[j,k]])).convert('RGB')
                        
                        rs = max(1,tile.width/100, tile.height/100)
                        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
                        full_image.paste(tile, (int((4000 - 100)*tx_imgs[indicesI[j,k]]), int((3000 - 100)*ty_imgs[indicesI[j,k]])), mask = tile.convert('RGBA'))
                        
                    filename = 'tsne-Euclid-Retrieval20-Image'+str(i)+'-'+str(j)+'.png'
                    full_image.save(filename)
                
                
                
                
                
    def evalRetrieval(self, model, embedded, printTop = 1):
        
        model.eval()
        
        test_imgs = []
        test_sents = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    test_imgs.append(images['filename'])
                    sents = images['sentids']
                    for sent in sents:
                        test_sents.append(sent)
                    
        '''
        with open('filenames.txt', 'a+') as f:
            for filename in test_imgs:
                string = filename + '\n'
                f.write(string)
        
        with open('sentids.txt', 'a+') as f:
            for sent in test_sents:
                string = str(sent) + '\n'
                f.write(string)
        '''    
        ranksI = np.zeros((1,5*model.dataset.ntest))
        ranksT = np.zeros((1,model.dataset.ntest)) 
        
               
        z_imgs_embedded = embedded[0:1000,:]
        z_attrs_embedded = embedded[1000:6000,:]
        
        tx_imgs = z_imgs_embedded[:,0]
        ty_imgs = z_imgs_embedded[:,1]
        tx_attrs = z_attrs_embedded[:,0]
        ty_attrs = z_attrs_embedded[:,1]
        
        minx = min(np.min(tx_imgs), np.min(tx_attrs))
        maxx = max(np.max(tx_imgs), np.max(tx_attrs))
        miny = min(np.min(ty_imgs), np.min(ty_attrs))
        maxy = max(np.max(ty_imgs), np.max(ty_attrs))
        
        tx_imgs = (tx_imgs-minx)/(maxx-minx)
        ty_imgs = (ty_imgs-miny)/(maxy-miny)
        tx_attrs =(tx_attrs-minx)/(maxx-minx)
        ty_attrs =(ty_attrs-miny)/(maxy-miny)
        
        txt = 'RetrievalTSNE-previouscode-'+str(printTop)+'.txt'
        
        with open(txt, 'w') as txtfile:
            
            indixI = []
            indixT = []
            
            ins = []
            for i in range(0, model.dataset.ntest):
                '''
                mu_img = model.gallery_imgs_z[i,:].unsqueeze(0)  
                mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
                
                distancesI = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                #distancesI = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                distancesT = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
                #distancesT = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                '''
                
                embd_img = np.zeros((1,2))
                embd_img[:,0] = tx_imgs[i]
                embd_img[:,1] = ty_imgs[i]
                
                embd_att = np.zeros((5,2))
                
                for ii in range(0,len(embd_att)):
                    embd_att[ii,0] = tx_attrs[5*i + ii]
                    embd_att[ii,1] = ty_attrs[5*i + ii]
                    
                gal_img = np.zeros((len(tx_imgs),2))
                gal_attrs = np.zeros((len(tx_attrs), 2))
                
                for ii in range(0, len(gal_img)):
                    gal_img[ii,0] = tx_imgs[ii]
                    gal_img[ii,1] = ty_imgs[ii]
                    
                    for iii in range(0,5):
                        gal_attrs[5*ii + iii, 0] = tx_attrs[5*ii + iii]
                        gal_attrs[5*ii + iii, 1] = ty_attrs[5*ii + iii]
                                        
                
                distancesI = distance.cdist(embd_att, gal_img, metric = 'cosine')
                distancesT = distance.cdist(embd_img, gal_attrs, metric = 'cosine')
                
                indicesI = np.argsort(distancesI)
                indicesT = np.argsort(distancesT[0,:])
                
                indixI.append(indicesI)
                indixT.append(indicesT)
                
                ins.append(i)
                
                
                
                for z in range(0,5):
                    if len(indicesI[z] == i) != 0:
                        ranksI[:,(5*i) + z] = np.where(indicesI[z] == i)[0][0]
                    else:
                        ranksI[:,(5*i) + z] = 1000
                        
                    full_image = Image.new('RGBA', (4000, 3000))
                    
                    for k in range(0,20):
                        tile = Image.open(os.path.join(self.path,'images', test_imgs[indicesI[z,k]])).convert('RGB')
                        
                        rs = max(1,tile.width/100, tile.height/100)
                        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
                        full_image.paste(tile, (int((4000 - 100)*tx_imgs[indicesI[z,k]]), int((3000 - 100)*ty_imgs[indicesI[z,k]])), mask = tile.convert('RGBA'))
                        
                    filename = 'tsne-prevcode-RetrievalTSNE20-Caption'+str(i)+'-'+str(z)+'.png'
                    print(filename)
                    full_image.save(filename)                    
                    
                    if ranksI[:,(5*i) + z] < printTop:
                        
                        que = 'QUERY'+ str((5*i)+z)+'\n'
                        txtfile.write(que)
                        
                        captions = self.bbdd[int(i)]['sentences']
                        caption = captions[z]
                        string = caption['raw'] + '\n'
                        txtfile.write(string)
                        
                        for j in range(0,10):
                            im = self.bbdd[indicesI[z][j]]['filename']
                        
                            img = Image.open(os.path.join(self.path, 'images', im))   
                        
                            fn = im
                            fn_str = fn + '\n'
                            txtfile.write(fn_str)
                            img.save(fn)
                        
                        txtfile.write('\n')
                        
                
                
                if len(np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))) != 0:
                    ranksT[:,i] = np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))[0][0]
                else:
                    ranksT[:,i] = 1000
                
                full_image = Image.new('RGBA', (4000, 3000))
                
                
                for k in range(0,20):
                    tile = Image.open(os.path.join(self.path,'images', test_imgs[int(indicesT[k]//5)])).convert('RGB')
                        
                    rs = max(1,tile.width/100, tile.height/100)
                    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
                    full_image.paste(tile, (int((4000 - 100)*tx_imgs[int(indicesT[k]//5)]), int((3000 - 100)*ty_imgs[int(indicesT[k]//5)])), mask = tile.convert('RGBA'))
                        
                filename = 'tsne-prevcode-RetrievalTSNE20-Image'+str(i)+'.png'
                print(filename)
                full_image.save(filename)              
                    
                if ranksT[:,i] < printTop:
                    im = self.bbdd[int(i)]['filename']
                    
                    img = Image.open(os.path.join(self.path, 'images', im))               
                    
                    
                    fn = im
                    fn_str = fn + '\n'
                    txtfile.write(fn_str)
                    #print(fn)
                    img.save(fn)
                    
                    for j in range(0,10):
                        captions = self.bbdd[int(indicesT[j]//5)]['sentences']
                        
                        
                        caption = captions[int(indicesT[j]%5)]
                        
                        string = caption['raw'] +'\n'
                        txtfile.write(string)
                        #print(caption['raw'])
                    
                    txtfile.write('\n')
                
                
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
        
        #Printar metrics
        print('Evaluation Metrics for image retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsI[0], metricsI[1], metricsI[2], metricsI[3], metricsI[4], metricsI[5], metricsI[6]))
        print('Evaluation Metrics for caption retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsT[0], metricsT[1], metricsT[2], metricsT[3], metricsT[4], metricsT[5], metricsT[6]))
        
    
        txtfile.close()
        
        test_imgs = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    test_imgs.append(images['filename'])
        
        '''
        train_samples = np.vstack((model.gallery_imgs_z.clone().cpu().detach(),model.gallery_attrs_z.clone().cpu().detach()))
        tsne = TSNE(n_components = 2)
        #tsne = tsne.fit(model.gallery_imgs_z.clone().cpu().detach())
        #tsne = tsne.fit(model.gallery_attrs_z.clone().cpu().detach())
        
        #z_imgs_embedded = tsne.fit_transform(model.gallery_imgs_z.clone().cpu().detach())
        embedded = tsne.fit_transform(train_samples)
        
        z_imgs_embedded = embedded[0:1000,:]
        z_attrs_embedded = embedded[1000:6000,:]
        
        
        tx_imgs = z_imgs_embedded[:,0]
        ty_imgs = z_imgs_embedded[:,1]
        tx_attrs = z_attrs_embedded[:,0]
        ty_attrs = z_attrs_embedded[:,1]
        
        minx = min(np.min(tx_imgs), np.min(tx_attrs))
        maxx = max(np.max(tx_imgs), np.max(tx_attrs))
        miny = min(np.min(ty_imgs), np.min(ty_attrs))
        maxy = max(np.max(ty_imgs), np.max(ty_attrs))
        
        
        tx_imgs = (tx_imgs-np.min(tx_imgs))/(np.max(tx_imgs)-np.min(tx_imgs))
        ty_imgs = (ty_imgs-np.min(ty_imgs))/(np.max(ty_imgs)-np.min(ty_imgs))
        tx_attrs =(tx_attrs-np.min(tx_attrs))/(np.max(tx_attrs)-np.min(tx_attrs))
        ty_attrs =(ty_attrs-np.min(ty_attrs))/(np.max(ty_attrs)-np.min(ty_attrs))
        
        
        tx_imgs = (tx_imgs-minx)/(maxx-minx)
        ty_imgs = (ty_imgs-miny)/(maxy-miny)
        tx_attrs =(tx_attrs-minx)/(maxx-minx)
        ty_attrs =(ty_attrs-miny)/(maxy-miny)
        '''
        
        
        
        full_image = Image.new('RGBA', (4000, 3000))
        full_captions = Image.new('RGBA', (4000,3000))
        
        for z in range(0, model.dataset.ntest):
                
            tile = Image.open(os.path.join(self.path,'images', test_imgs[z])).convert('RGB')
                            
            rs = max(1,tile.width/100, tile.height/100)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
            full_image.paste(tile, (int((4000 - 100)*tx_imgs[z]), int((3000 - 100)*ty_imgs[z])), mask = tile.convert('RGBA'))
            #full_captions.paste(tile,(int((4000-100)*tx_imgs[z]), int((3000-100)*ty_imgs[z])), mask = tile.convert('RGBA'))
            for i in range(0,5):
                caption_x = tx_attrs[5*z+i]
                caption_y = ty_attrs[5*z+i]

                full_captions.paste(tile, (int((4000-100)*caption_x), int((3000-100)*caption_y)), mask = tile.convert('RGBA'))                           
                
                  
        filename = 'Image-t-sne-plot-prevcode-images.png'
        full_image.save(filename)
        plt.clf()
        
        filename = 'Image-t-sne-plot-prevcode-captions.png'
        full_captions.save(filename)
        plt.clf
            
    
    def captionRetrieval(self,model):
        
        model.eval()
        
        test_imgs = []
        test_sents = []
        test_s = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    test_imgs.append(images['filename'])
                    sents = images['sentids']
                    test_sents.append(sents)
                    
                    for sent in sents:
                        test_s.append(int(sent))
        
        ranked = []
        ranks = np.zeros((1,5000))
        for i in range(0, len(test_imgs)):
            mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
            
            for j in range(0,len(test_sents[i])):
                que_att = mu_att[j:j+1,:]
                
                distancesT = distance.cdist(que_att.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
                indicesT = np.argsort(distancesT)
                
                sentis = test_sents[i].copy()
                sent = sentis.copy()
                del sent[j:j+1]
                
                ranks[:,(5*i)+j] = 5000
                
                
                for z in sent: 
                    ks = 0
                    for s in test_s:
                        if s == z:
                            idxs = ks
                        
                        ks += 1
                    
                    print(idxs)
                    if len(np.where(indicesT[0,:] == idxs)[0]) != 0:
                        ranking = np.where(indicesT[0,:] == idxs)[0][0] 
                        ranking -= 1
                        ranked.append(ranking)
                    else:
                        ranking = 5000
                        ranking -= 1
                        ranked.append(ranking)
                    
                    if ranking < ranks[:,(5*i)+j]:
                        ranks[:,(5*i)+j] = ranking
                
                
        r1im = 100.0 * len(np.where(ranks < 1)[1]) / len(ranks[0,:])
        r5im = 100.0 * len(np.where(ranks< 5)[1]) / len(ranks[0,:])
        r10im = 100.0 * len(np.where(ranks < 10)[1]) / len(ranks[0,:])
        r50im = 100.0 * len(np.where(ranks < 50)[1]) / len(ranks[0,:])
        r100im = 100.0 * len(np.where(ranks < 100)[1]) / len(ranks[0,:])
        
        medrI = np.floor(np.median(ranks)) + 1
        meanrI = ranks.mean() + 1
        
        metricsI = [r1im, r5im, r10im, r50im, r100im, medrI, meanrI]
        
        print('Evaluation Metrics for caption-to-caption retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsI[0], metricsI[1], metricsI[2], metricsI[3], metricsI[4], metricsI[5], metricsI[6]))
        
        ranked = np.array(ranked)
        
        r1im = 100.0 * len(np.where(ranked < 1)[0]) / len(ranked)
        r5im = 100.0 * len(np.where(ranked< 5)[0]) / len(ranked)
        r10im = 100.0 * len(np.where(ranked < 10)[0]) / len(ranked)
        r50im = 100.0 * len(np.where(ranked < 50)[0]) / len(ranked)
        r100im = 100.0 * len(np.where(ranked < 100)[0]) / len(ranked)
        
        medrI = np.floor(np.median(ranked)) + 1
        meanrI = ranked.mean() + 1
        
        metricsI = [r1im, r5im, r10im, r50im, r100im, medrI, meanrI]
        
        print('Evaluation Metrics for caption-to-caption retrieval (one positive)')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsI[0], metricsI[1], metricsI[2], metricsI[3], metricsI[4], metricsI[5], metricsI[6]))
                       
        
        
    #Seleccionar els items de la classe en concret.
    #Obtenir resultats quantitatius i qualitatius.
    def evalRetrievalClass(self, model, cluster = 'dog', printTop = 1):

        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder
        
        path = os.path.join(project_directory,'data','flickr30k')
        
        self.test_imgs = []
        attr_filename = os.path.join(path,'dataset.json')
        with open(attr_filename) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    self.test_imgs.append(images['imgid'])

        if cluster == 'dog':
            cluster_idxs = [34,1072,1723,1833,2124,3008,3097,4343,4510,5160,5171,5240,5330,5343,5441,5724,5867,5947,5970,6078,6250,6334,7739,7750, \
                            7912,8084,8185,8225,8486,8654,8766,8872,8940,9039,9736,10577,11234,11239,11423,11618,11747,12356,12402,12927,12994,13135,13276,13516,\
                                13785, 13990, 14332, 14434, 14736, 14832, 14989, 15020, 15186, 15650, 17822, 17823,18499, 18979, 18998, 21900, 22317, 22419, 22548,\
                                22594, 23128, 23280, 24846, 27391, 29632, 30660]
        
        elif cluster == 'horse':
            cluster_idxs = [6376, 12508, 12969, 13276, 14721, 20154, 22258, 23713, 24517, 24521, 25456, 26536, 28389, 28662, 30664]
        
        elif cluster == 'water':
            cluster_idxs = [304,754,828,973,1306,1343,1576,2044,2103,2260,2265,3078,3245,3724,4313,4879,5581,6250,6258,6733,7153,7549,7581,7909,\
                            7996, 8427,8631, 8674, 10010, 10469, 11086, 11624, 12184, 12441, 13417, 13629, 15047, 15556, 16347, 17177, 18310,\
                            18532, 19624, 20934, 21067, 21127, 21635, 22098, 22145, 22678, 22953, 23115, 23560, 25918, 26142, 26160, 26393, 26682,\
                            27250, 28480, 28966, 28978, 29185, 29754, 29772, 30283, 30943]
        
        elif cluster == 'snow':
            cluster_idxs = [89,133,2906,3451,4317,11423,11963,12706,12994,13648,15592,17630, 18666, 19162, 25583, 25795, 25986, 28246]
        
        elif cluster == 'bike':
            cluster_idxs = [734,906,2139,3087,9088,9598,9803, 10920, 12053, 14117, 14164, 14596, 14786, 15448, 16048, 16191, 17903, 17963, 18425,\
                            20507, 21696, 22098, 22678, 22860, 23205, 24181, 24315, 24425, 24495, 26015, 26084, 28240, 30105, 30201, 30494, 30843]
        
        elif cluster == 'bicicle':
            cluster_idxs = [734,906,2139,3087,9088,9598,9803,10920, 12053, 14117, 14164, 14596, 14786, 15448, 16048, 16191, 18425, 20507, 21696,\
                            22098, 22678, 22860, 23205, 24181, 24315, 24425, 24495, 26015, 26084, 28240, 30105, 30201, 30494, 30843]
        
        elif cluster == 'motorcicle':
            cluster_idxs = [15448, 17903, 17963]
            
        elif cluster == 'group_people':
            cluster_idxs = [97, 133,222,228,259,318,395,434,661,734,748, 760,828,1030,1117,1202,  1305,  1310,  1417,  1480,  1576,  1594,  1610,  1646,\
                             1771,  2044,  2054,  2112,  2113,  2130,  2131,  2139,  2319,  2330,  2523,  2600,  2678,  2906,  3078,  3087,  3130,  3245,\
                            3412,  3648,  3724,  4041,  4663,  4881,  4981,  5016,  5261,  5327,  5580,  5581,  5743,  5964,  6192,  6258,  6283,  6306,\
                            6486,  6549,  6645,  6796,  6830,  6921,  7153, 7170,  7492,  7631,  7702,  7921, 7973,  7986,  8015,  8023,  8054,  8070,\
                            8079,  8199,  8362,  8364,  8561,  8708,  8831,  8903,  8993,  9065,  9241,  9328 , 9492,  9531,  9609,  9673,  9797,  9885,\
                            9888,  9911,  9959,  9978, 10088, 10106, 10242, 10287, 10330, 10376, 10379, 10629, 10691, 10692, 10693, 10732, 10858, 11124,\
                            11219, 11278, 11283, 11363, 11516, 11624, 11656, 11890, 11906, 11939, 11979, 12030, 12084, 12133, 12212, 12366, 12402, 12417,\
                            12508, 12542, 12757, 12853, 12969, 13170, 13566, 13650, 13724, 13744, 13837, 14048, 14072, 14134, 14231, 14316, 14453, 14490,\
                            14495, 14626, 14690, 14796, 14941, 14974, 15067, 15077, 15176, 15358, 15374, 15556, 15759, 15774, 15834, 15878, 15974, 16004,\
                            16010, 16018, 16063, 16070, 16090, 16381, 16455, 16600, 16638, 16888, 16989, 16996, 17031, 17121, 17208, 17250, 17346, 17376,\
                            17468, 17573, 17630, 17652, 17655, 17673, 17708, 17733, 17750, 17772, 17933, 17940, 17945, 17978, 18146, 18250, 18310, 18418,\
                            18419, 18426, 18584, 18731, 18764, 18998, 19027, 19075, 19174, 19435, 19478, 19523, 19724, 19738, 19772, 19834, 19867, 20071,\
                            20139, 20448, 20459, 20891, 20898, 20934, 20946, 20981, 21105, 21121, 21127, 21219, 21222, 21301, 21423, 21498, 21510, 21580,\
                            21597, 21635, 21725, 21764, 21778, 21996, 22042, 22078, 22115, 22118, 22254, 22383, 22440, 22476, 22613, 22651, 22751, 22776,\
                            22835, 22847, 22865, 22870, 22886, 22953, 22968, 22994, 22998, 23006, 23070, 23098, 23099, 23107, 23115, 23144, 23240, 23252,\
                            23346, 23351, 23502, 23509, 23617, 23790, 23863, 23986, 24071, 24095, 24145, 24181, 24277, 24315, 24327, 24476, 24517, 24535,\
                            24806, 24829, 24836, 24863, 24974, 25069, 25133, 25316, 25479, 25516, 25525, 25554, 25626, 25637, 25763, 25820, 26049, 26084,\
                            26094, 26370, 26393, 26407, 26411, 26440, 26615, 26645, 26806, 26843, 26906, 26982, 26984, 27048, 27238, 27240, 27303, 27321,\
                            27495, 27682, 27726, 27804, 27863, 27988, 28005, 28026, 28041, 28067, 28068, 28139, 28328, 28389, 28470, 28570, 28582, 28661,\
                            28767, 28873, 28893, 28955, 29043, 29084, 29108, 29185, 29197, 29352, 29364, 29608, 29636, 29754, 29791, 29944, 29952, 30065,\
                            30283, 30480, 30503, 30549, 30652, 30781, 30893]
            
        elif cluster == 'woman':
            cluster_idxs = [51, 124, 219, 228, 279, 318, 340, 395, 472, 487, 539, 565, 661, 748, 856, 887, 917, 973,1030 , 1358 , 1646,  1821,  1895 , 1983,\
                                1984,  1987,  2054,  2103 , 2113 , 2131,  2180,  2410,  2429,  2458,  2523,  2615, 2788,  2848,  3078 , 3130,  3476,  3539 , 3703 ,\
                                3771,  3834 , 4367,  4501,  4663, 4748,  4816,  4882,  4942,  4981,  5016,  5056,  5240,  5541 , 5580 , 5743,  6002,  6067 , 6068,\
                                6300,  6661,  7170,  7263 , 7581 , 7597,  7702,  7986 , 7996 , 8015, 8023 , 8054 , 8070 , 8079 , 8274 , 8362 , 8412,  8427 , 8554 , 8708 , 8740,  8982,\
                                8993,  9223,  9241,  9274,  9308 , 9506 , 9598 , 9673 , 9776 , 9885 ,10010, 10088,\
                                10287, 10379, 10620, 10925, 11137, 11283, 11363, 11374, 11514, 11897, 11906, 11939,\
                                11979, 12178, 12245, 12293, 12305, 12427, 13085, 13388, 13417, 13543, 13770, 13916,\
                                14042, 14134, 14316, 14333, 14337, 14479, 14690, 14773, 14796, 14841, 14974, 15047,\
                                15138, 15374, 15412, 15457, 15588, 15809, 15834, 15974, 16004 ,16090, 16191, 16199,\
                                16416, 16421, 16458, 16532, 16600, 16638, 16715, 16871, 16989, 17031, 17037, 17194,\
                                17338, 17468, 17483, 17603, 17708, 17714, 17731, 17733, 17750, 17963, 18028 ,18425,\
                                18426, 18584, 18723, 18764, 19005, 19037, 19075, 19122, 19277, 19348, 19373, 19384,\
                                19471, 19609, 19624, 19738, 19772, 19857, 19929, 20016, 20139, 20229, 20297, 20302,\
                                20457, 20562, 20592, 20640, 20740, 20832, 20898, 21187, 21301, 21580, 21597, 21799,\
                                22115, 22118, 22163, 22334, 22339, 22389, 22613, 22755, 22835, 22865, 22886, 22892,\
                                22953, 22994, 23006, 23070, 23115, 23252, 23351, 23502, 23509, 23617, 23756, 23850,\
                                24071, 24095, 24169, 24777, 24836, 24863, 25058, 25188, 25376, 25494, 25525, 25583,\
                                25626, 25753, 25820, 25952, 25987, 26039, 26142, 26352, 26393, 26407, 26598, 26618,\
                                26946, 26969, 26982, 26984, 27025, 27168, 27250, 27391, 27457, 27733, 27830, 28066,\
                                28068, 28246, 28328, 28470, 28885, 29508, 29830, 29862, 29916, 29952, 30111, 30201,\
                                30674, 30781, 30843, 30943]
        elif cluster == 'man':
            cluster_idxs = [25,    51,    97 ,  124 ,  134  , 137  , 199  , 219 ,  222 ,  246 ,  304 ,  340,\
                            395,   520 ,  539 ,  615  , 670 ,  734 ,  754 ,  828 ,  972,  1023 , 1030  ,1072,\
                            1117,  1202,  1206,  1284,  1305,  1306,  1390,  1417,  1480,  1576,  1610,  1771,\
                            1821,  1981,  2044,  2130,  2131,  2139,  2161,  2260,  2265,  2319,  2412,  2429,\
                            2433,  2523,  2551,  2572,  2600,  2615,  2690,  2803,  2851,  2909,  2955,  2956,\
                            3109,  3130,  3162,  3245,  3412,  3437,  3500,  3648,  3724,  3774,  3826,  3951,\
                            3972,  4007,  4027,  4028,  4041,  4313,  4402,  4452,  4556,  4562,  4686,  4801,\
                            4881,  4902,  4942,  5261,  5327,  5580,  5581,  5595,  5618,  5621,  5701,  5760,\
                            5763,  5767,  5807,  5835,  5854,  5881,  5989,  6005,  6031,  6127,  6242,  6250,\
                            6306,  6486,  6532,  6549,  6645,  6694,  7153,  7492,  7507,  7542,  7549,  7584,\
                            7639,  7682,  7874,  7881,  7909,  7921,  7962,  8015,  8054,  8070,  8175,  8267,\
                            8274,  8280,  8322,  8337,  8364,  8519,  8528,  8561,  8611,  8631,  8674,  8776,\
                            8811,  8870,  8993,  9013,  9058,  9065,  9074,  9088,  9223,  9242,  9266,  9342,\
                            9354,  9502,  9519,  9531,  9572,  9579,  9797,  9803,  9885,  9888,  9908,  9911,\
                            9959,  9978,  9979, 10005, 10010, 10097, 10106, 10124, 10171, 10235, 10239, 10242,\
                            10330, 10379, 10390, 10490, 10499, 10575, 10663, 10666, 10691, 10693, 10788, 10920,\
                            11124, 11207, 11234, 11283, 11400, 11466, 11514, 11516, 11656, 11890, 11914, 11979,\
                            12030, 12058, 12077, 12133, 12212, 12366, 12375, 12417, 12475, 12508, 12542, 12757,\
                            12782, 12813, 12853, 12865, 12902, 12947, 12949, 12969, 13170, 13197, 13302, 13388,\
                            13566, 13580, 13629, 13650, 13724, 13744, 13837, 13992, 14042, 14048, 14164, 14221,\
                            14231, 14260, 14453, 14490, 14495, 14558, 14570, 14582, 14596, 14599, 14626, 14786,\
                            14796, 14941, 15004, 15061, 15067, 15077, 15176, 15219, 15309, 15358, 15410, 15448,\
                            15674, 15759, 15774, 15809, 15875, 15878, 15974, 16049, 16063, 16070, 16124, 16347,\
                            16381, 16455, 16631, 16739, 16773, 16810, 16880, 16938, 16992, 16996, 17045, 17057,\
                            17071, 17109, 17121, 17177, 17208, 17250, 17347, 17376, 17468, 17483, 17497, 17510,\
                            17533, 17573, 17615, 17655, 17669, 17903, 17940, 17963, 17978, 17983, 17997, 18061,\
                            18146, 18155, 18250, 18418, 18419, 18426, 18448, 18532, 18731, 18820, 18994, 18998,\
                            19071, 19162, 19247, 19263, 19354, 19356, 19435, 19478, 19545, 19555, 19639, 19643,\
                            19738, 19772, 19830, 19838, 19896, 19910, 20005, 20071, 20103, 20240, 20248, 20390,\
                            20448, 20507, 20656, 20682, 20709, 20832, 20858, 20891, 20932, 20981, 21105, 21121,\
                            21187, 21219, 21222, 21301, 21335, 21423, 21510, 21544, 21597, 21635, 21696, 21725,\
                            21778, 21996, 22042, 22067, 22078, 22145, 22193, 22223, 22254, 22317, 22341, 22387,\
                            22425, 22440, 22476, 22478, 22554, 22678, 22751, 22755, 22776, 22778, 22788, 22865,\
                            22870, 22956, 22968, 23065, 23070, 23091, 23098, 23099, 23107, 23164, 23240, 23252,\
                            23345, 23351, 23509, 23560, 23686, 23724, 23758, 23790, 23827, 23863, 23957, 23965,\
                            23986, 24043, 24181, 24244, 24271, 24277, 24327, 24350, 24425, 24457, 24476, 24495,\
                            24517, 24545, 24624, 24806, 24829, 24974, 25069, 25133, 25149, 25235, 25315, 25316,\
                            25494, 25516, 25525, 25541, 25554, 25637, 25763, 25776, 25819, 25918, 25986, 26015,\
                            26049, 26050, 26094, 26133, 26256, 26366, 26370, 26377, 26407, 26419, 26440, 26446,\
                            26512, 26615, 26645, 26682, 26697, 26806, 26843, 26846, 26906, 26922, 26969, 26998,\
                            27048, 27238, 27240, 27303, 27457, 27495, 27636, 27671, 27682, 27726, 27804, 27817,\
                            27863, 27988, 28005, 28026, 28041, 28063, 28113, 28123, 28139, 28142, 28240, 28389,\
                            28470, 28605, 28694, 28767, 28819, 28827, 28873, 28923, 28955, 28966, 29043, 29084,\
                            29177, 29197, 29250, 29301, 29323, 29352, 29357, 29364, 29380, 29480, 29489, 29608,\
                            29629, 29636, 29791, 30283, 30384, 30456, 30480, 30503, 30542, 30557, 30619, 30652,\
                            30665, 30674, 30893, 30942]
                
        print(len(cluster_idxs))
        
        model.eval()
        
        ranksI = np.zeros((1,5*len(cluster_idxs)))
        ranksT = np.zeros((1,len(cluster_idxs)))    
        
        txt = 'Retrieval-'+str(printTop)+'-'+cluster+'.txt'
        
        with open(txt, 'w') as txtfile:
            k = 0
            for idx in cluster_idxs:
                
                i = self.test_imgs.index(idx)
                mu_img = model.gallery_imgs_z[i,:].unsqueeze(0)  
                mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
                
                distancesI = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                distancesT = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
                
                indicesI = np.argsort(distancesI)
                indicesT = np.argsort(distancesT[0,:])
                
                print(idx)
                print(i)
                
                for z in range(0,5):
                    print(z)
                    if len(indicesI[z] == i) != 0:
                        ranksI[:,(5*k) + z] = np.where(indicesI[z] == i)[0][0]
                    else:
                        ranksI[:,(5*k) + z] = 1000
                    
                    if ranksI[:,(5*k) + z] < printTop:
                        
                        que = 'QUERY'+ str((5*i)+z)
                        txtfile.write(que)
                        
                        captions = self.bbdd[int(i)]['sentences']
                        caption = captions[z]
                        
                        txtfile.write(caption['raw'])
                        
                        for j in range(0,10):
                            im = self.bbdd[indicesI[z][j]]['filename']
                        
                            img = Image.open(os.path.join(self.path, 'images', im))   
                        
                            fn = 'r'+str(j)+'-'+str((5*i)+z)+'.png'
                            txtfile.write(fn)
                            img.save(fn)
                        
                
                
                if len(np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))) != 0:
                    ranksT[:,k] = np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))[0][0]
                else:
                    ranksT[:,k] = 1000
                    
                if ranksT[:,k] < printTop:
                    im = self.bbdd[int(i)]['filename']
                    
                    img = Image.open(os.path.join(self.path, 'images', im))               
                    
                    
                    fn = 'query'+str(i)+'.png'
                    txtfile.write(fn)
                    #print(fn)
                    img.save(fn)
                    
                    for j in range(0,10):
                        captions = self.bbdd[indicesT[j] // 5]['sentences']
                        caption = captions[indicesT[j]%5]
                        
                        txtfile.write(caption['raw'])
                        #print(caption['raw'])
                k += 1
        
        txtfile.close()
        
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
        
        print('Evaluation Metrics for image retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsI[0], metricsI[1], metricsI[2], metricsI[3], metricsI[4], metricsI[5], metricsI[6]))
        print('Evaluation Metrics for caption retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsT[0], metricsT[1], metricsT[2], metricsT[3], metricsT[4], metricsT[5], metricsT[6]))
        
        
    def distsSpace(self, model):
        
        cluster_idxs = [34,1072,1723,1833,2124,3008,3097,4343,4510,5160,5171,5240,5330,5343,5441,5724,5867,5947,5970,6078,6250,6334,7739,7750, \
                            7912,8084,8185,8225,8486,8654,8766,8872,8940,9039,9736,10577,11234,11239,11423,11618,11747,12356,12402,12927,12994,13135,13276,13516,\
                                13785, 13990, 14332, 14434, 14736, 14832, 14989, 15020, 15186, 15650, 17822, 17823,18499, 18979, 18998, 21900, 22317, 22419, 22548,\
                                22594, 23128, 23280, 24846, 27391, 29632, 30660]
        
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder
        
        path = os.path.join(project_directory,'data','flickr30k')
        
        self.test_imgs = []
        attr_filename = os.path.join(path,'dataset.json')
        with open(attr_filename) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    self.test_imgs.append(images['imgid'])
                    
        z_imgs = []
        z_attrs = []
        
        j = 0
        for idx in cluster_idxs:
                
            i = self.test_imgs.index(idx)
            mu_img = model.gallery_imgs_z[i,:].unsqueeze(0)  
            mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
            
            if j == 0:
                z_imgs = mu_img.cpu()
                z_attrs = mu_att.cpu()
            else:
                z_imgs = torch.cat((z_imgs.cpu(),mu_img.cpu()), dim = 0).cpu()
                z_attrs = torch.cat((z_attrs.cpu(), mu_att.cpu()), dim = 0).cpu()
            
            
            j += 1
        
        print(z_imgs[0:1,:].shape)
        print(z_attrs[0:1,:].shape)
        distancesI = distance.cdist(z_imgs[0:1,:].detach().numpy(), z_imgs.detach().numpy(), 'cosine')
        distancesT = distance.cdist(z_attrs[0:1,:].detach().numpy(), z_attrs.detach().numpy(), 'cosine')
        
        print(distancesI[0,:])
        print(distancesT[0,:])
        
    
    def i2t(self,images, captions, embedded, model, npts=None, measure='cosine', return_ranks=False):
        """
        Images->Text (Image Annotation)
        Images: (5N, K) matrix of images
        Captions: (5N, K) matrix of captions
        """
        
        printTop = 10
        
        test_imgs = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for imag in image_list:
                if imag['split'] == 'test':
                    test_imgs.append(imag['filename'])
        
        z_imgs_embedded = embedded[0:1000,:]
        z_attrs_embedded = embedded[1000:6000,:]
        
        tx_imgs = z_imgs_embedded[:,0]
        ty_imgs = z_imgs_embedded[:,1]
        tx_attrs = z_attrs_embedded[:,0]
        ty_attrs = z_attrs_embedded[:,1]
        
        minx = min(np.min(tx_imgs), np.min(tx_attrs))
        maxx = max(np.max(tx_imgs), np.max(tx_attrs))
        miny = min(np.min(ty_imgs), np.min(ty_attrs))
        maxy = max(np.max(ty_imgs), np.max(ty_attrs))
        
        tx_imgs = (tx_imgs-minx)/(maxx-minx)
        ty_imgs = (ty_imgs-miny)/(maxy-miny)
        tx_attrs =(tx_attrs-minx)/(maxx-minx)
        ty_attrs =(ty_attrs-miny)/(maxy-miny)
        
        images = normalize(images, axis = 1, norm = 'l2')
        #im_vars = normalize(im_vars, axis = 1, norm = 'l2')
        captions = normalize(captions, axis = 1, norm = 'l2')
        #captions_vars = normalize(captions_vars,axis = 1, norm = 'l2')
        
        def order_sim(im, s):
            """Order embeddings similarity measure $max(0, s-im)$
            """
            YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
                   - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
            score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
            return score
        
        def kl_divergence(p_mu, p_var, q_mu, q_var):
            return (0.5 * torch.sum(1 + p_var - p_mu.pow(2) - p_var.exp()))+ (0.5 * torch.sum(1 + q_var - q_mu.pow(2) - q_var.exp()))

        if npts is None:
            npts = int(images.shape[0] / 5)
            
        
        index_list = []
    
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        
        txt = 'Retrieval-newcode-I2T-'+str(printTop)+'.txt'
        
        with open(txt, 'w') as txtfile:
            for index in range(npts):
                            
                # Get query image
                im = images[5 * index].reshape(1, images.shape[1])
                #im_var = im_vars[5 * index].reshape(1,images.shape[1])
        
                # Compute scores
                if measure == 'order':
                    bs = 100
                    if index % bs == 0:
                        mx = min(images.shape[0], 5 * (index + bs))
                        im2 = images[5 * index:mx:5]
                        d2 = order_sim(torch.Tensor(im2).cuda(),
                                       torch.Tensor(captions).cuda())
                        d2 = d2.cpu().numpy()
                    d = d2[index % bs]
                else:
                    d = np.dot(im, captions.T).flatten()
                    #for i in range(0, len(d)):
                    #    d[i] = kl_divergence(im, im_var, captions[i,:], captions_vars[i,:])
                    
                inds = np.argsort(d)[::-1]
                index_list.append(inds[0])
                
                
                full_image = Image.new('RGBA', (4000, 3000))
                        
                for k in range(0,20):
                    tile = Image.open(os.path.join(self.path,'images', test_imgs[int(inds[k]//5)])).convert('RGB')
                            
                    rs = max(1,tile.width/100, tile.height/100)
                    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                    
                    full_image.paste(tile, (int((4000 - 100)*tx_imgs[int(inds[k]//5)]), int((3000 - 100)*ty_imgs[int(inds[k]//5)])), mask = tile.convert('RGBA'))
                            
                filename = 'tsne-newcode-cosine-Retrieval20-Image'+str(index)+'.png'
                full_image.save(filename)
                
        
                # Score
                rank = 1e20
                for i in range(5 * index, 5 * index + 5, 1):
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank
                top1[index] = inds[0]
                
                if rank < 10:
                    im = self.bbdd[int(index)]['filename']
                    
                    img = Image.open(os.path.join(self.path, 'images', im))               
                    
                    
                    fn = im
                    fn_str = fn + '\n'
                    txtfile.write(fn_str)
                    #print(fn)
                    img.save(fn)
                    
                    for j in range(0,10):
                        captis = self.bbdd[int(inds[j]//5)]['sentences']
                        
                        
                        caption = captis[int(inds[j]%5)]
                        
                        string = caption['raw'] +'\n'
                        txtfile.write(string)
                        #print(caption['raw'])
                    
                    txtfile.write('\n')
    
        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        
        metrics = [r1, r5, r10, medr, meanr]
        
        
        
        full_image = Image.new('RGBA', (4000, 3000))
        full_captions = Image.new('RGBA', (4000,3000))
        
        for z in range(0, model.dataset.ntest):
                
            tile = Image.open(os.path.join(self.path,'images', test_imgs[z])).convert('RGB')
                            
            rs = max(1,tile.width/100, tile.height/100)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
            full_image.paste(tile, (int((4000 - 100)*tx_imgs[z]), int((3000 - 100)*ty_imgs[z])), mask = tile.convert('RGBA'))
            #full_captions.paste(tile,(int((4000-100)*tx_imgs[z]), int((3000-100)*ty_imgs[z])), mask = tile.convert('RGBA'))
            for i in range(0,5):
                caption_x = tx_attrs[5*z+i]
                caption_y = ty_attrs[5*z+i]

                full_captions.paste(tile, (int((4000-100)*caption_x), int((3000-100)*caption_y)), mask = tile.convert('RGBA'))                           
                
                  
        filename = 'Image-t-sne-cosine-plot-newcode-images.png'
        full_image.save(filename)
        plt.clf()
        
        filename = 'Image-t-sne-cosine-plot-newcode-captions.png'
        full_captions.save(filename)
        plt.clf
        
        return metrics
    
    def t2i(self,images, captions, embedded, npts=None, measure='cosine', return_ranks=False):
        """
        Text->Images (Image Search)
        Images: (5N, K) matrix of images
        Captions: (5N, K) matrix of captions
        """
        
        printTop = 10
        
        test_imgs = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for imag in image_list:
                if imag['split'] == 'test':
                    test_imgs.append(imag['filename'])
        
        
        z_imgs_embedded = embedded[0:1000,:]
        z_attrs_embedded = embedded[1000:6000,:]
        
        tx_imgs = z_imgs_embedded[:,0]
        ty_imgs = z_imgs_embedded[:,1]
        tx_attrs = z_attrs_embedded[:,0]
        ty_attrs = z_attrs_embedded[:,1]
        
        minx = min(np.min(tx_imgs), np.min(tx_attrs))
        maxx = max(np.max(tx_imgs), np.max(tx_attrs))
        miny = min(np.min(ty_imgs), np.min(ty_attrs))
        maxy = max(np.max(ty_imgs), np.max(ty_attrs))
        
        tx_imgs = (tx_imgs-minx)/(maxx-minx)
        ty_imgs = (ty_imgs-miny)/(maxy-miny)
        tx_attrs =(tx_attrs-minx)/(maxx-minx)
        ty_attrs =(ty_attrs-miny)/(maxy-miny)
        
        images = normalize(images, axis = 1, norm = 'l2')
        captions = normalize(captions, axis = 1, norm = 'l2')
        
        def order_sim(im, s):
            """Order embeddings similarity measure $max(0, s-im)$
            """
            YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
                   - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
            score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
            return score
        
        def kl_divergence(p_mu, p_var, q_mu, q_var):
            return (0.5 * torch.sum(1 + p_var - p_mu.pow(2) - p_var.exp())) + (0.5 * torch.sum(1 + q_var - q_mu.pow(2) - q_var.exp()))

        if npts is None:
            npts = int(images.shape[0] / 5)
        
        ims = np.array([images[i] for i in range(0, len(images), 5)])
    
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
        
        txt = 'Retrieval-newcode-T2I-'+str(printTop)+'.txt'
        
        with open(txt, 'w') as txtfile:
            for index in range(npts):
        
                # Get query captions
                queries = captions[5 * index:5 * index + 5]
        
                # Compute scores
                if measure == 'order':
                    bs = 100
                    if 5 * index % bs == 0:
                        mx = min(captions.shape[0], 5 * index + bs)
                        q2 = captions[5 * index:mx]
                        d2 = order_sim(torch.Tensor(ims).cuda(),
                                       torch.Tensor(q2).cuda())
                        d2 = d2.cpu().numpy()
        
                    d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
                else:
                    d = np.dot(queries, ims.T)
                    #for i in range(0, len(d)):
                    #    d[i] = kl_divergence(queries[int(i%5)], queries_vars[int(i%5)], images[i], im_vars[i])
                     
                inds = np.zeros(d.shape)
                for i in range(len(inds)):
                    inds[i] = np.argsort(d[i])[::-1]
                    ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
                    top1[5 * index + i] = inds[i][0]
                    
                    full_image = Image.new('RGBA', (4000, 3000))
                    
                    for k in range(0,20):
                        tile = Image.open(os.path.join(self.path,'images', test_imgs[int(inds[i][k])])).convert('RGB')
                        
                        rs = max(1,tile.width/100, tile.height/100)
                        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
                        full_image.paste(tile, (int((4000 - 100)*tx_imgs[int(inds[i][k])]), int((3000 - 100)*ty_imgs[int(inds[i][int(k)])])), mask = tile.convert('RGBA'))
                        
                    filename = 'tsne-newcode-cosine-Retrieval20-Caption'+str(index)+'-'+str(i)+'.png'
                    full_image.save(filename)        
                    
                    if ranks[5*index + i] < 10:
                        que = 'QUERY'+ str((5*index)+i)+'\n'
                        txtfile.write(que)
                                
                        captis = self.bbdd[int(index)]['sentences']
                        caption = captis[i]
                        string = caption['raw'] + '\n'
                        txtfile.write(string)
                                
                        for j in range(0,10):
                            im = self.bbdd[int(inds[i][j]//5)]['filename']
                                
                            img = Image.open(os.path.join(self.path, 'images', im))   
                                
                            fn = im
                            fn_str = fn + '\n'
                            txtfile.write(fn_str)
                            img.save(fn)
                                
                        txtfile.write('\n')
        
        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        
        metrics = [r1, r5, r10, medr, meanr]
        return metrics
                
                
                
                
            
            
     
