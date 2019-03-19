import cv2
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def show(img):
    #print(img.shape)
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)
    
def show_gray(img):
    #print(img.shape)
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)
    
def save_gray(img, path):
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    print('Image saved to ', path)
    pilImg.save(path)

    
def predict(model, img, epoch, path):
    #to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
    #im = to_tensor(img)
    #show(im)
    inp = img.unsqueeze(0)
    #print(inp.size())
    out = model(inp)
    map_out = out.cpu().data.squeeze(0)
    #show_gray(map_out)
    
    new_path = path + str(epoch) + ".png"
    save_gray(map_out, new_path)
    

