import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from scipy import misc
import os
import os.path
import numpy as np
import random
import time

def default_loader(impath, gtpath):
    img = Image.open(impath).convert('RGB')
    gt = Image.open(gtpath).convert("L")
    return img,gt
    
def default_list_reader(fileList):
    imgList = []
    gtList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, gtPath = line.strip().split('\t')
            imgList.append(imgPath)
            gtList.append(gtPath)
    return imgList, gtList

class ImageLoader(data.Dataset):
    def __init__(self, root, fileList, data_type = " ", transform = None,  list_reader = default_list_reader, loader = default_loader):
        self.root = root
        self.imgList, self.gtList = list_reader(fileList)
        self.transform = T.ToTensor()
        self.loader = loader # assigning the object
        self.data_type = data_type

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        gtpath = self.gtList[index]
        #img = self.loader(os.path.join(self.root, imgPath),'img') # actual function call of loader
        #gt = self.loader(os.path.join(self.root, gtPath),'gt')
        img,gt = self.loader(os.path.join(self.root,imgpath),os.path.join(self.root,gtpath))
        if self.data_type == 'train':
            if (random.random() > 0.5):
                img = F.hflip(img)
                gt = F.hflip(gt)          

        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
            #img =  torch.from_numpy(np.array(img)).permute(2,0,1) 
            #gt =  torch.from_numpy(np.array(gt)).unsqueeze(0)
        
        return img, gt

    def __len__(self):
        return len(self.imgList)
