import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

HOME_DIR = "/data/Goutam_Kelam/SALGAN/SaliconDataset/"
NEW_PAT	H = "/data/Goutam_Kelam/SALGAN/NewSalGan/"

# Path to Dataset
org_path_imgs = HOME_DIR+"images/"
org_path_maps_train = HOME_DIR+"maps/train/"
org_path_maps_val = HOME_DIR+"maps/val/"
org_path_maps_test = HOME_DIR+"maps/test/"

# Path to processed images
path_imgs_train= NEW_PATH+"Images/train/" 
path_imgs_test = NEW_PATH+"Images/test/" 
path_imgs_val = NEW_PATH+"Images/val/" 
path_maps_train = NEW_PATH+"Maps/train" 
path_maps_test = NEW_PATH+"Maps/test" 
path_maps_val = NEW_PATH+"Maps/val" 

# Desired size of processed image
Out_Size = (256,192)

# Creating Directory (if not existing)
if not os.path.exists(path_imgs_train):
    os.makedirs(path_imgs_train)
if not os.path.exists(path_imgs_test):
    os.makedirs(path_imgs_test)
if not os.path.exists(path_imgs_val):
    os.makedirs(path_imgs_val)
if not os.path.exists(path_maps_train):
    os.makedirs(path_maps_train)
if not os.path.exists(path_maps_test):
    os.makedirs(path_maps_test)
if not os.path.exists(path_maps_val):
    os.makedirs(path_maps_val) 

list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(org_path_imgs, 'train/*'))]
print("Total number of training images: ",len(list_img_files))
#print(list_img_files)
#assert False
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(org_path_imgs+"train/", curr_file + '.jpg')
    try:
        imageResized = cv2.resize(cv2.imread(full_img_path), Out_Size, interpolation=cv2.INTER_AREA)
            
        full_map_path = os.path.join(org_path_maps_train, curr_file + '.png')
        mapResized = cv2.resize(cv2.imread(full_map_path), Out_Size, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(os.path.join(path_imgs_train, curr_file + '.png'), imageResized)
        cv2.imwrite(os.path.join(path_maps_train, curr_file + '.png'), mapResized)
    except:
        print('Error')
    
list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(org_path_imgs, 'val/*'))]
print("\nTotal number of validation images: ",len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(org_path_imgs+"val/", curr_file + '.jpg')
    imageResized = cv2.resize(cv2.imread(full_img_path), Out_Size, interpolation=cv2.INTER_AREA)
        
    full_map_path = os.path.join(org_path_maps_val, curr_file + '.png')
    mapResized = cv2.resize(cv2.imread(full_map_path), Out_Size, interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(os.path.join(path_imgs_val, curr_file + '.png'), imageResized)
    cv2.imwrite(os.path.join(path_maps_val, curr_file + '.png'), mapResized)
   

list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(org_path_imgs, 'test/*'))]
print("\nTotal number of test images: ",len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(org_path_imgs+"test/", curr_file + '.jpg')
    imageResized = cv2.resize(cv2.imread(full_img_path), Out_Size, interpolation=cv2.INTER_AREA)
    
    full_map_path = os.path.join(org_path_maps_test, curr_file + '.png')
    mapResized = cv2.resize(cv2.imread(full_map_path), Out_Size, interpolation=cv2.INTER_AREA)        
    
    cv2.imwrite(os.path.join(path_imgs_test, curr_file + '.png'), imageResized)
    cv2.imwrite(os.path.join(path_maps_test, curr_file + '.png'), imageResized)
    

print('Done resizing images.')
