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

animals_imgs = []
dog_imgs = []
horse_imgs = []
bird_imgs = []

sea_imgs = []
ocean_imgs = []
river_imgs = []
lake_imgs = []
water_imgs = []

beach_imgs = []

vehicles_imgs = []
bike_imgs = []
bicicle_imgs = []
motorcicle_imgs = []
car_imgs = []
boat_imgs = []

snow_imgs = []

man_imgs = []
men_imgs = []
boy_imgs = []
male_imgs = []
woman_imgs = []
women_imgs = []
girl_imgs = []
female_imgs = []
group_people_imgs = []
children_imgs = []
people_imgs = []

person_imgs = []
child_imgs = []
baby_imgs = []
crowd_imgs = []
background_imgs = []

clothes_imgs = []
shirt_imgs = []
hat_imgs = []
glasses_imgs = []

asian_imgs = []

nature_imgs = []
tree_imgs = []
flower_imgs = []
forest_imgs = []
landscape_imgs = []
sky_imgs = []
cloud_imgs = []

sunset_imgs = []
night_imgs = []
indoor_imgs = []

food_imgs = []

chair_imgs = []
table_imgs = []

graffitti_imgs = []

sign_imgs = []

ball_imgs = []

portrait_imgs = []

urbanscene_imgs = []
urban_imgs = []
city_imgs = []
buildings_imgs = []
building_imgs = []
architecture_imgs = []
bridge_imgs = []
road_imgs = []
street_imgs = []

house_imgs = []
home_imgs = []
house_home_imgs = []

young_imgs = []
old_imgs = []
small_imgs = []
big_imgs = []
little_imgs = []
plural_imgs = []

dark_imgs = []
blond_imgs = []

white_imgs = []
black_imgs = []
gray_imgs = []
blue_imgs = []
red_imgs = []
green_imgs = []
brown_imgs = []
yellow_imgs = []

attr_filename = os.path.join(path,'dataset.json')
with open(attr_filename) as f:
    json_data = json.loads(f.read())
    image_list = json_data['images']
    
    for images in image_list:
        if images['split'] == 'test':
            for sent in images['sentences']:
                if 'dog' in sent['tokens']:
                    dog_imgs.append(images['imgid'])
                    animals_imgs.append(images['imgid'])
                elif 'horse' in sent['tokens']:
                    horse_imgs.append(images['imgid'])
                    animals_imgs.append(images['imgid'])
                elif 'bird' in sent['tokens']:
                    bird_imgs.append(images['imgid'])
                    animals_imgs.append(images['imgid'])
                elif 'animal' in sent['tokens']:
                    animals_imgs.append(images['imgid'])
                    
                elif 'water' in sent['tokens']:
                    water_imgs.append(images['imgid'])
                elif 'sea' in sent['tokens']:
                    sea_imgs.append(images['imgid'])
                    water_imgs.append(images['imgid'])
                elif 'ocean' in sent['tokens']:
                    ocean_imgs.append(images['imgid'])
                    water_imgs.append(images['imgid'])
                elif 'river' in sent['tokens']:
                    river_imgs.append(images['imgid'])
                    water_imgs.append(images['imgid'])
                elif 'lake' in sent['tokens']:
                    lake_imgs.append(images['imgid'])
                    water_imgs.append(images['imgid'])
                
                elif 'beach' in sent['tokens']:
                    beach_imgs.append(images['imgid'])
                    
                elif 'bike' in sent['tokens']:
                    bike_imgs.append(images['imgid'])
                    bicicle_imgs.append(images['imgid'])
                    vehicles_imgs.append(images['imgid'])
                elif 'motorbike' in sent['tokens']:
                    bike_imgs.append(images['imgid'])
                    motorcicle_imgs.append(images['imgid'])
                    vehicles_imgs.append(images['imgid'])
                    
                elif 'snow' in sent['tokens']:
                    snow_imgs.append(images['imgid'])
                    
                elif 'man' in sent['tokens']:
                    man_imgs.append(images['imgid'])
                    male_imgs.append(images['imgid'])
                elif 'woman' in sent['tokens']:
                    woman_imgs.append(images['imgid'])
                    female_imgs.append(images['imgid'])
                elif 'boy' in sent['tokens']:
                    boy_imgs.append(images['imgid'])
                    male_imgs.append(images['imgid'])
                elif 'girl' in sent['tokens']:
                    girl_imgs.append(images['imgid'])
                    female_imgs.append(images['imgid'])
                elif 'men' in sent['tokens']:
                    male_imgs.append(images['imgid'])
                    men_imgs.append(images['imgid'])
                    group_people_imgs.append(images['imgid'])
                elif 'women' in sent['tokens']:
                    female_imgs.append(images['imgid'])
                    women_imgs.append(images['imgid'])
                    group_people_imgs.append(images['imgid'])
                elif 'children' in sent['tokens']:
                    children_imgs.append(images['imgid'])
                    group_people_imgs.append(images['imgid'])
                elif 'people' in sent['tokens']:
                    people_imgs.append(images['imgid'])
                    group_people_imgs.append(images['imgid'])
                elif 'person' in sent['tokens']:
                    person_imgs.append(images['imgid'])
                elif 'child' in sent['tokens']:
                    child_imgs.append(images['imgid'])
                elif 'baby' in sent['tokens']:
                    baby_imgs.append(images['imgid'])
                elif 'crowd' in sent['tokens']:
                    crowd_imgs.append(images['imgid'])
                    
                elif 'background' in sent['tokens']:
                    background_imgs.append(images['imgid'])
                    
                elif 'car' in sent['tokens']:
                    car_imgs.append(images['imgid'])
                    vehicles_imgs.append(images['imgid'])
                elif 'boat' in sent['tokens']:
                    boat_imgs.append(images['imgid'])
                    vehicles_imgs.append(images['imgid'])
                    
                elif 'shirt' in sent['tokens']:
                    shirt_imgs.append(images['imgid'])
                    clothes_imgs.append(images['imgid'])
                elif 'hat' in sent['tokens']:
                    hat_imgs.append(images['imgid'])
                    clothes_imgs.append(images['imgid'])
                elif 'glasses' in sent['tokens']:
                    glasses_imgs.append(images['imgid'])
                    clothes_imgs.append(images['imgid'])
                
                elif 'Asian' in sent['tokens']:
                    asian_imgs.append(images['imgid'])
                
                elif 'nature' in sent['tokens']:
                    nature_imgs.append(images['imgid'])
                elif 'tree' in sent['tokens']:
                    tree_imgs.append(images['imgid'])
                    nature_imgs.append(images['imgid'])
                elif 'forest' in sent['tokens']:
                    forest_imgs.append(images['imgid'])
                    nature_imgs.append(images['imgid'])
                elif 'flower' in sent['tokens']:
                    flower_imgs.append(images['imgid'])
                    nature_imgs.append(images['imgid'])
                elif 'landscape' in sent['tokens']:
                    landscape_imgs.append(images['imgid'])
                    nature_imgs.append(images['imgid'])
                elif 'sky' in sent['tokens']:
                    sky_imgs.append(images['imgid'])
                elif 'clouds' in sent['tokens']:
                    cloud_imgs.append(images['imgid'])
                    sky_imgs.append(images['imgid'])
                    
                elif 'sunset' in sent['tokens']:
                    sunset_imgs.append(images['imgid'])
                elif 'night' in sent['tokens']:
                    night_imgs.append(images['imgid'])
                elif 'indoor' in sent['tokens']:
                    indoor_imgs.append(images['imgid'])
                elif 'food' in sent['tokens']:
                    food_imgs.append(images['imgid'])
                
                elif 'chair' in sent['tokens']:
                    chair_imgs.append(images['imgid'])
                elif 'table' in sent['tokens']:
                    table_imgs.append(images['imgid'])
                
                elif 'graffiti' in sent['tokens']:
                    graffitti_imgs.append(images['imgid'])
                elif 'sign' in sent['tokens']:
                    sign_imgs.append(images['imgid'])
                elif 'portrait' in sent['tokens']:
                    portrait_imgs.append(images['imgid'])
                elif 'ball' in sent['tokens']:
                    ball_imgs.append(images['imgid'])
                elif 'building' in sent['tokens']:
                    building_imgs.append(images['imgid'])
                    buildings_imgs.append(images['imgid'])
                    urbanscene_imgs.append(images['imgid'])
                elif 'architecture' in sent['tokens']:
                    architecture_imgs.append(images['imgid'])
                    buildings_imgs.append(images['imgid'])
                    urbanscene_imgs.append(images['imgid'])
                elif 'house' in sent['tokens']:
                    house_imgs.append(images['imgid'])
                    house_home_imgs.append(images['imgid'])
                elif 'home' in sent['tokens']:
                    home_imgs.append(images['imgid'])
                    house_home_imgs.append(images['imgid'])
                elif 'city' in sent['tokens']:
                    city_imgs.append(images['imgid'])
                    urbanscene_imgs.append(images['imgid'])
                elif 'urban' in sent['tokens']:
                    urban_imgs.append(images['imgid'])
                    urbanscene_imgs.append(images['imgid'])
                elif 'bridge' in sent['tokens']:
                    bridge_imgs.append(images['imgid'])
                    urbanscene_imgs.append(images['imgid'])
                elif 'road' in sent['tokens']:
                    road_imgs.append(images['imgid'])
                    urbanscene_imgs.append(images['imgid'])
                elif 'street' in sent['tokens']:
                    urbanscene_imgs.append(images['imgid'])
                    street_imgs.append(images['imgid'])
                
                elif 'young' in sent['tokens']:
                    young_imgs.append(images['imgid'])
                elif 'old' in sent['tokens']:
                    old_imgs.append(images['imgid'])
                elif 'older' in sent['tokens']:
                    old_imgs.append(images['imgid'])
                elif 'small' in sent['tokens']:
                    small_imgs.append(images['imgid'])
                elif 'big' in sent['tokens']:
                    big_imgs.append(images['imgid'])
                elif 'large' in sent['tokens']:
                    big_imgs.append(images['imgid'])
                elif 'little' in sent['tokens']:
                    little_imgs.append(images['imgid'])
                elif 'several' in sent['tokens']:
                    plural_imgs.append(images['imgid'])
                elif 'many' in sent['tokens']:
                    plural_imgs.append(images['imgid'])
                elif 'dark' in sent['tokens']:
                    dark_imgs.append(images['imgid'])
                elif 'blond' in sent['tokens']:
                    blond_imgs.append(images['imgid'])
                elif 'white' in sent['tokens']:
                    white_imgs.append(images['imgid'])
                elif 'black' in sent['tokens']:
                    black_imgs.append(images['imgid'])
                elif 'gray' in sent['tokens']:
                    gray_imgs.append(images['imgid'])
                elif 'blue' in sent['tokens']:
                    blue_imgs.append(images['imgid'])
                elif 'red' in sent['tokens']:
                    red_imgs.append(images['imgid'])
                elif 'green' in sent['tokens']:
                    green_imgs.append(images['imgid'])
                elif 'yellow' in sent['tokens']:
                    yellow_imgs.append(images['imgid'])
                elif 'brown' in sent['tokens']:
                    brown_imgs.append(images['imgid'])
                
                    
                
print('Animal images: ', np.unique(np.array(animals_imgs)))
print('Dog images: ', np.unique(np.array(dog_imgs)))
print('Horse images: ',np.unique(np.array(horse_imgs)))
print('Bird images: ',np.unique(np.array(bird_imgs)))
print('Water images: ',np.unique(np.array(water_imgs)))
print('Sea images: ',np.unique(np.array(sea_imgs)))
print('Ocean images: ',np.unique(np.array(ocean_imgs)))
print('River images: ',np.unique(np.array(river_imgs)))
print('Lake images: ',np.unique(np.array(lake_imgs)))
print('Beach images: ',np.unique(np.array(beach_imgs)))
print('Vehicles images: ',np.unique(np.array(vehicles_imgs)))
print('Bike images: ',np.unique(np.array(bike_imgs)))
print('Bicicle images: ',np.unique(np.array(bicicle_imgs)))
print('Motorcicle images: ',np.unique(np.array(motorcicle_imgs)))
print('Car images: ',np.unique(np.array(car_imgs)))
print('Boat images: ',np.unique(np.array(boat_imgs)))
print('Snow images: ',np.unique(np.array(snow_imgs)))