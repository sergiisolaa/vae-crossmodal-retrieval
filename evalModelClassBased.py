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
    
    def evalRetrieval(self, model, printTop = 1):
        
        model.eval()
        
        ranksI = np.zeros((1,model.dataset.ntest))
        ranksT = np.zeros((1,5*model.dataset.ntest))    
        
        txt = 'Retrieval-'+str(printTop)+'.txt'
        
        with open(txt, 'w') as txtfile:
            
            for i in range(0, model.dataset.ntest):
                
                mu_img = model.gallery_imgs_z[i,:].unsqueeze(0)  
                mu_att = model.gallery_attrs_z[5*i:5*i + 5,:]
                
                #distancesI = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                distancesI = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                #distancesT = distance.cdist(mu_img.cpu().detach().numpy(), model.gallery_attrs_z.cpu().detach().numpy(), 'cosine')
                distancesT = distance.cdist(mu_att.cpu().detach().numpy(), model.gallery_imgs_z.cpu().detach().numpy(), 'cosine')
                
                indicesI = np.argsort(distancesI)
                indicesT = np.argsort(distancesT)
                
                '''
                for z in range(0,5):
                    if len(indicesT[z] == i) != 0:
                        ranksT[:,(5*i) + z] = np.where(indicesT[z] == i)[0][0]
                    else:
                        ranksT[:,(5*i) + z] = 1000
                    
                    if ranksT[:,(5*i) + z] < printTop:
                        
                        que = 'QUERY'+ str((5*i)+z)
                        txtfile.write(que)
                        
                        captions = self.bbdd[int(i)]['sentences']
                        caption = captions[z]
                        
                        txtfile.write(caption['raw'])
                        
                        for j in range(0,10):
                            im = self.bbdd[indicesT[z][j]]['filename']
                        
                            img = Image.open(os.path.join(self.path, 'images', im))   
                        
                            fn = 'r'+str(j)+'-'+str((5*i)+z)+'.png'
                            txtfile.write(fn)
                            img.save(fn)
                        
                
                
                if len(np.where((indicesI >= 5*i) & (indicesI <= ((5*i) + 4)))) != 0:
                    ranksI[:,i] = np.where((indicesI >= 5*i) & (indicesI <= ((5*i) + 4)))[0][0]
                else:
                    ranksI[:,i] = 1000
                    
                if ranksI[:,i] < printTop:
                    im = self.bbdd[int(i)]['filename']
                    
                    img = Image.open(os.path.join(self.path, 'images', im))               
                    
                    
                    fn = 'query'+str(i)+'.png'
                    txtfile.write(fn)
                    #print(fn)
                    img.save(fn)
                    
                    for j in range(0,10):
                        captions = self.bbdd[indicesI[j] // 5]['sentences']
                        caption = captions[indicesI[j]%5]
                        
                        txtfile.write(caption['raw'])
                        #print(caption['raw'])
                '''
                '''
                for z in range(0,5):
                    if len(indicesI[z] == i) != 0:
                        ranksT[:,(5*i) + z] = np.where(indicesT[z] == i)[0][0]
                    else:
                        ranksT[:,(5*i) + z] = 1000
                    
                    if ranksT[:,(5*i) + z] < printTop:
                        
                        que = 'QUERY'+ str((5*i)+z)
                        txtfile.write(que)
                        
                        captions = self.bbdd[int(i)]['sentences']
                        caption = captions[z]
                        
                        txtfile.write(caption['raw'])
                        
                        for j in range(0,10):
                            im = self.bbdd[indicesT[z][j]]['filename']
                        
                            img = Image.open(os.path.join(self.path, 'images', im))   
                        
                            fn = 'r'+str(j)+'-'+str((5*i)+z)+'.png'
                            txtfile.write(fn)
                            img.save(fn)
                '''       
                '''
                if len(indicesI == i) != 0:
                    ranksI[:,i] = np.where(indicesI == i)[0][0]
                else:
                    ranksI[:,i] = 1000
                
                if len(np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))) != 0:
                    ranksT[:,i] = np.where((indicesT >= 5*i) & (indicesT <= ((5*i) + 4)))[1][0]
                else:
                    ranksT[:,i] = 1000
                    
                if ranksT[:,i] < printTop:
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
        '''
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
        
        #Printar metrics
        print('Evaluation Metrics for image retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsI[0], metricsI[1], metricsI[2], metricsI[3], metricsI[4], metricsI[5], metricsI[6]))
        print('Evaluation Metrics for caption retrieval')
        print("R@1: {}, R@5: {}, R@10: {}, R@50: {}, R@100: {}, MEDR: {}, MEANR: {}".format(metricsT[0], metricsT[1], metricsT[2], metricsT[3], metricsT[4], metricsT[5], metricsT[6]))
        '''
    
        txtfile.close()
        
        test_imgs = []
        with open(os.path.join(self.path,'dataset.json')) as f:
            json_data = json.loads(f.read())
            image_list = json_data['images']
            
            for images in image_list:
                if images['split'] == 'test':
                    test_imgs.append(images['filename'])
        
        
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
        
        '''
        tx_imgs = (tx_imgs-np.min(tx_imgs))/(np.max(tx_imgs)-np.min(tx_imgs))
        ty_imgs = (ty_imgs-np.min(ty_imgs))/(np.max(ty_imgs)-np.min(ty_imgs))
        tx_attrs =(tx_attrs-np.min(tx_attrs))/(np.max(tx_attrs)-np.min(tx_attrs))
        ty_attrs =(ty_attrs-np.min(ty_attrs))/(np.max(ty_attrs)-np.min(ty_attrs))
        '''
        
        tx_imgs = (tx_imgs-minx)/(maxx-minx)
        ty_imgs = (ty_imgs-miny)/(maxy-miny)
        tx_attrs =(tx_attrs-minx)/(maxx-minx)
        ty_attrs =(ty_attrs-miny)/(maxy-miny)
        
        full_image = Image.new('RGBA', (4000, 3000))
        
        
        for z in range(0, model.dataset.ntest):
            
            full_captions = Image.new('RGBA', (4000,3000))
                
            tile = Image.open(os.path.join(self.path,'images', test_imgs[z])).convert('RGB')
                
            imshow(tile)
                
            rs = max(1,tile.width/100, tile.height/100)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                
            full_image.paste(tile, (int((4000 - 100)*tx_imgs[z]), int((3000 - 100)*ty_imgs[z])), mask = tile.convert('RGBA'))
            full_captions.paste(tile, (int((4000-100)*tx_imgs[z]), int((3000-100)*ty_imgs[z])), mask = tile.convert('RGBA'))
            for i in range(0,5):
                caption_x = tx_attrs[5*z+i]
                caption_y = ty_attrs[5*z+i]

                full_captions.paste(tile, (int((4000-100)*caption_x), int((3000-100)*caption_y)), mask = tile.convert('RGBA'))     
                
            filename = 'captionsTSNE+Image-'+str(z)+'Autoencoder.png'
            full_captions.save(filename)
            plt.clf          
                
        '''          
        filename = 'Image-t-sne-plot-Retrieval-imagesNorm-fitCorrect.png'
        full_image.save(filename)
        plt.clf()
        
        filename = 'Image-t-sne-plot-Retrieval-captionsNorm-fitCorrect.png'
        full_captions.save(filename)
        plt.clf
        '''
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
        
        ranksI = np.ones((1,5*len(cluster_idxs)))*1000
        ranksT = np.ones((1,len(cluster_idxs)))*1000    
        
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
                
                for z in range(0,5):
                    
                    for jdx in cluster_idxs:
                        j = self.test_imgs.index(jdx)
                        if len(np.where(indicesI[z] == j)[0]) != 0:
                            if np.where(indicesI[z] == j)[0][0] < ranksI[:,(5*k)+z]:
                                ranksI[:,(5*k) + z] = np.where(indicesI[z] == j)[0][0]
                    
                    print(ranksI[:,(5*k)+z])
                    
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
                        
                
                
                for jdx in cluster_idxs:
                    j = self.test_imgs.index(jdx)
                    if len(np.where((indicesT >= 5*j) & (indicesT <= ((5*j) + 4)))[0]) != 0:          
                        if np.where((indicesT >= 5*j) & (indicesT <= ((5*j) + 4)))[0][0] < ranksT[:,k]:                            
                            ranksT[:,k] = np.where((indicesT >= 5*j) & (indicesT <= ((5*j) + 4)))[0][0]
                
                print(ranksT[:,k])
                    
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
        
        
        
        
        
        
 
