
#

import glob
import os
import torchvision.transforms as transforms
import numpy as np
import torch

from generator import Generator
from discriminator import Discriminator
from PIL import Image, ImageOps


def show(img): # Display rgb tensor image
    pilImg = img
    pilImg.show()

def show_gray(img): # Display grayscale tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    pilImg.show()

def show_img_from_path(imgPath):
    pilImg = Image.open(imgPath)
    pilImg.show()

def predict(model, img):
    to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
    im = to_tensor(img).cuda()
    inp = im.unsqueeze(0)
    #inp = im
    out = model(inp)
    map_out = out.cpu().data.squeeze(0)
    return map_out


pathToResizedImagesVal = './Images/val/*'
pathToResizedMapsVal = './Maps/val/*'

list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(pathToResizedImagesVal)]
print(len(list_img))
#print(list_img)
model = Generator()
pretrained_dict = torch.load('./Newgenerator.pkl')
model.load_state_dict(pretrained_dict)
if torch.cuda.is_available():
    model.cuda()
#print(model)
#imgName = list_img[700] + ".png"
#imgName = "COCO_val2014_ (11).png"
#print(imgName)
#img_path = pathToResizedImagesVal + imgName
#map_ground_truth = pathToResizedMapsVal + imgName
#img_path = "./Images/val/" + imgName
#map_ground_truth = "./Maps/val/" + imgName
#print(img_path)
#print(map_ground_truth)
#assert False
img_path = "./sai.png"
img = Image.open(img_path).convert("RGB")
#img = ImageOps.fit(img,(256,192),Image.ANTIALIAS)
sal_predicted = predict(model,img)
show(img)
show_gray(sal_predicted)
show_img_from_path(map_ground_truth)

to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
im = to_tensor(img)
inp_d = torch.cat((im,sal_predicted),0)
inp_d.unsqueeze_(0)
print(inp_d.shape)
inp_d = inp_d.cuda()

model = Discriminator()
pretrained_dict = torch.load('./Newdiscriminator.pkl')
model.load_state_dict(pretrained_dict)
if torch.cuda.is_available():
    model.cuda()
output = model(inp_d)
print(output)
