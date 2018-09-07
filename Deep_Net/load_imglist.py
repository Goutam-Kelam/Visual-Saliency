import torch
import torch.utils.data as data
from PIL import Image, ImageOps
from scipy import misc
import os
import os.path
import numpy as np
import random
import torchvision.transforms.functional as TF

def default_loader(path, otype = 'img'):
    if otype == 'img':
        img = Image.open(path)
        img = img.convert('RGB')
        img = ImageOps.fit(img, (180,180), Image.ANTIALIAS)
        img = torch.FloatTensor(np.array(img))
        mean = torch.FloatTensor([118.0,110.0,100.0])
        img = img - mean.view(1,1,-1)
        img = (img*0.0039).permute(2,0,1)
        return img
    elif otype == 'gt':
        gt = Image.open(path).convert('L')
        gt = ImageOps.fit(gt, (176, 176), Image.ANTIALIAS)
        gt = torch.FloatTensor(np.array(gt))
        mean = torch.FloatTensor([27])
        gt = gt - mean
        gt = (gt*0.0039).unsqueeze(2).permute(2,0,1)
        return gt
    
def default_list_reader(fileList):
    imgList = []
    gtList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, gtPath = line.strip().split('\t')
            imgList.append(imgPath)
            gtList.append(gtPath)
    return imgList, gtList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform = None, list_reader = default_list_reader,  
                 loader = default_loader,type_of_data = None):
        self.root = root
        self.imgList, self.gtList = list_reader(fileList)
        self.transform = transform
        self.loader = loader # assigning the object
        self.type_of_data = type_of_data 

    def __getitem__(self, index):
        imgPath = self.imgList[index]
        gtPath = self.gtList[index]
        img = self.loader(os.path.join(self.root, imgPath),'img') # actual function call of loader
        gt = self.loader(os.path.join(self.root, gtPath),'gt')
        
        """if self.type_of_data == "train": # for randomly flipping both img and gt horizontally 
            if random.random()>0.5:
                img = TF.hflip(img)#.convert("RGB")
                gt = TF.hflip(gt)#.convert("L")
            
        if self.transform is not None:
            img = self.transform(img)  # Used ToTensor to scale to [0,1]
            gt = self.transform(gt)
        
        img = (2.*img)-1 # For scaling to [-1,1]
        gt = (2.*gt)-1
        """

        return img, gt

    def __len__(self):
        return len(self.imgList)
