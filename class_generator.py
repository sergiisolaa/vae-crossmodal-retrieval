# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:25:01 2020

@author: PC
"""

import os
from pathlib import Path
import json
import numpy as np

folder = str(Path(os.getcwd()))
if folder[-5:] == 'model':
    project_directory = Path(os.getcwd()).parent
else:
    project_directory = folder
        
path = os.path.join(project_directory,'data','flickr30k')

dog_imgs = []
horse_imgs = []
water_imgs = []
bike_imgs = []
bicicle_imgs = []
motorcicle_imgs = []
snow_imgs = []
man_imgs = []
woman_imgs = []
group_people_imgs = []

attr_filename = os.path.join(path,'dataset.json')
with open(attr_filename) as f:
    json_data = json.loads(f.read())
    image_list = json_data['images']
    
    for images in image_list:
        if images['split'] == 'test':
            for sent in images['sentences']:
                if 'dog' in sent['tokens']:
                    dog_imgs.append(images['imgid'])
                elif 'horse' in sent['tokens']:
                    horse_imgs.append(images['imgid'])
                elif 'water' in sent['tokens']:
                    water_imgs.append(images['imgid'])
                elif 'bike' in sent['tokens']:
                    bike_imgs.append(images['imgid'])
                    bicicle_imgs.append(images['imgid'])
                elif 'motorbike' in sent['tokens']:
                    bike_imgs.append(images['imgid'])
                    motorcicle_imgs.append(images['imgid'])
                elif 'snow' in sent['tokens']:
                    snow_imgs.append(images['imgid'])
                elif 'man' in sent['tokens']:
                    man_imgs.append(images['imgid'])
                elif 'woman' in sent['tokens']:
                    woman_imgs.append(images['imgid'])
                elif 'boy' in sent['tokens']:
                    man_imgs.append(images['imgid'])
                elif 'girl' in sent['tokens']:
                    woman_imgs.append(images['imgid'])
                elif 'men' in sent['tokens']:
                    man_imgs.append(images['imgid'])
                    group_people_imgs.append(images['imgid'])
                elif 'women' in sent['tokens']:
                    woman_imgs.append(images['imgid'])
                    group_people_imgs.append(images['imgid'])
                elif 'children' in sent['tokens']:
                    group_people_imgs.append(images['imgid'])
                elif 'people' in sent['tokens']:
                    group_people_imgs.append(images['imgid'])
    
print('Dog images: ', np.unique(np.array(dog_imgs)))
print('Horse images: ',np.unique(np.array(horse_imgs)))
print('Water images: ',np.unique(np.array(water_imgs)))
print(np.unique(np.array(bicicle_imgs)))
print(np.unique(np.array(bike_imgs)))
print(np.unique(np.array(motorcicle_imgs)))
print(np.unique(np.array(snow_imgs)))
print(np.unique(np.array(man_imgs)))
print(np.unique(np.array(woman_imgs)))
print(np.unique(np.array(group_people_imgs)))
    
