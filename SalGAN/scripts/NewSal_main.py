import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import  MultiStepLR
from tqdm import tqdm
import torch.optim as optim
 
import numpy as np
import random
import time
import argparse
import os
  
  
#added arguments.py
from Arguments import get_args
from PIL import Image

from my_data_loader import ImageLoader
from discriminator import Discriminator
from generator import Generator
from utils import *
  
global args #defining variables that can be accessed globally
global device #defining variables that can be accessed globally
  
#args = parser.parse_args()
args = get_args()
 
device = torch.device("cuda" if args.cuda else "cpu")
  
train_loader = torch.utils.data.DataLoader(
                  ImageLoader(root = args.root_path, fileList = args.train_list, data_type = "train",transform = None),
                  batch_size = args.batch_size,
                  shuffle = False,
                  num_workers = args.workers,
                  pin_memory = True)
  
val_loader = torch.utils.data.DataLoader(
                  ImageLoader(root = args.root_path, fileList = args.val_list, data_type = "val",transform = None),
                  batch_size = args.batch_size,
                  shuffle = False,
                  num_workers = args.workers,
                  pin_memory = True)

discriminator = Discriminator().to(device)
generator = Generator().to(device)
loss_function = nn.BCELoss()

lr = args.lr
to_tensor = transforms.ToTensor()
  
d_optim = optim.Adagrad(discriminator.parameters(), lr=lr)
g_optim = optim.Adagrad(generator.parameters(), lr=lr)

val_path = "./Images/val/COCO_val2014_ (1094).png"
DIR_TO_SAVE = "./New_generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)
validation_sample = Image.open(val_path).convert("RGB")
validation_sample = to_tensor(validation_sample).to(device)

# Main Loop

alpha = 0.05
start_time = time.time()
r_label = 1
f_label = 0

for epoch in tqdm(range(args.start_epoch+1,args.epochs+1)):

    for i, (img, gt) in enumerate(train_loader):
        img = img.to(device)
        gt = gt.to(device)
        real_label = torch.full((args.batch_size,),r_label,device=device)
        fake_label = torch.full((args.batch_size,),f_label,device=device)
        
        # Training Discriminator
       
        # Train with real
        d_optim.zero_grad()
        inp_d = torch.cat((img,gt),1)
        out = discriminator(inp_d).squeeze()
        d_real_loss = loss_function(out, real_label)
        d_real_loss.backward()
        D_x = out.mean().item()
 
        # Train with Fake
        fake_gt = generator(img)
        inp_d = torch.cat((img,fake_gt),1)
        out = discriminator(inp_d).squeeze()
        d_fake_loss = loss_function(out,fake_label)
        d_fake_loss.backward()
        D_G_z1 = out.mean().item()
        
        d_loss = d_real_loss + d_fake_loss

        d_optim.step()
       

        # Training Generator 

        g_optim.zero_grad()
        fake_gt = generator(img)
        inp_d = torch.cat((img,fake_gt),1)
        out = discriminator(inp_d).squeeze() #added Squeeze
         
        D_G_z2 = out.mean().item()       

        g_gen_loss = loss_function(fake_gt,gt)
        g_dis_loss = loss_function(out.detach(),real_label)
        g_loss = torch.sum(g_dis_loss + alpha*g_gen_loss)

        g_loss.backward()
        g_optim.step()

        if (i+1)%100 == 0:
            print(" Epoch: {}/{}, Step: {}/{}".format(epoch,args.epochs,i+1,len(train_loader)))
            print(" d_loss: {:.4f}, g_loss: {:.4f}".format(d_loss.item(), g_loss.item()))
            print(" D(x): {:.4f}, D(G(x)): {:.4f}/{:.4f}".format(D_x, D_G_z1,D_G_z2))
            print(" time: {:.4f}". format(time.time()-start_time))
            print("\n")

    # Save weights every 3th epoch
    if (epoch)%3 == 0:
        torch.save(generator.state_dict(), './Newgenerator.pkl')
        torch.save(discriminator.state_dict(), './Newdiscriminator.pkl')
    predict(generator, validation_sample, epoch, DIR_TO_SAVE)
  
torch.save(generator.state_dict(), './Newgenerator.pkl')
torch.save(discriminator.state_dict(), './Newdiscriminator.pkl')

print("training_over")
