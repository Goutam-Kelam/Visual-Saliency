
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
import os


#added arguments.py
from Arguments import get_args
from DeepNet_model import DeepNet
from load_imglist import ImageList

global iterations

iterations = 0


def Adjust_lr_scheduler(optimizer):
    #lr = init_lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*0.5


# In[4]:

def train(train_loader, model, criterion, optimizer, scheduler, epoch, step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()    
    
    global iterations

    model.train()

    end = time.time()
    for i, (ip, gt) in enumerate(train_loader):
        
        #if iterations%step == 0:
        #    Adjust_lr_scheduler(optimizer)

        data_time.update(time.time() - end)

        ip = ip.to(device)
        gt = gt.to(device)
        
        # compute output
        op = model(ip)
        loss = criterion(op, gt)

        # record loss
        losses.update(loss.item(), ip.size(0))
               
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #print(i)
        if i % args.print_freq == 0:
            #print('Iterations \t:',iterations)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time = batch_time,
                   data_time = data_time, loss = losses))

    iterations += 1
    if iterations % 2000 == 0:
        scheduler(optimizer)

    return losses.avg


# In[5]:

def validate(val_loader, model, criterion):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for i, (ip, gt) in enumerate(val_loader):
        ip = ip.to(device)
        gt = gt.to(device)
        
        with torch.no_grad():
            # compute output
            op = model(ip)
            loss = criterion(op, gt)
            
            # record loss
            losses.update(loss.item(), ip.size(0))
            
    print('\nValidation set: Average loss: {}\n'.format(losses.avg)) 
    return losses.avg




# In[8]:

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.summ = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.summ += val * n
        self.count += n
        self.avg = self.summ / self.count


# In[9]:

def save_checkpoint(state, filename):
    torch.save(state, filename)


# In[11]:


global args #defining variables that can be accessed globally
global device #defining variables that can be accessed globally

#args = parser.parse_args()
args = get_args()

device = torch.device("cuda" if args.cuda else "cpu")

# Instantiate model 
model = DeepNet(dataset = args.dataset).to(device)

# Instantiate optimizer and learning rate scheduler
optimizer = torch.optim.SGD(params = model.parameters(), 
                            lr = args.lr, 
                            momentum = args.momentum, 
                            weight_decay = args.weight_decay,
                            nesterov = True)

#lamda = lambda epoch: 0.995 ** epoch
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, 
#                                              lr_lambda = lamda)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
scheduler = Adjust_lr_scheduler#(optimizer)
# Instantiate loss function
criterion = nn.MSELoss()

# optionally resume from a checkpoint

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
'''
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

'''

#Instantiate image loaders
"""
train_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.train_list, type_of_data = "train",transform = transforms.Compose([ 
                                            #transforms.RandomHorizontalFlip(), 
                                            #transforms.ToTensor(),
                                            #transforms.Normalize(mean=[0.463,0.430,0.389])
                        ])),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = args.workers, 
                pin_memory = True)

val_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.val_list,type_of_data = "val",
                          transform = transforms.Compose([ 
                                          #transforms.RandomHorizontalFlip(),  
                                          #transforms.ToTensor(),
                                          #transforms.Normalize(mean=[0.459,0.428,0.389])
                        ])),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = args.workers, 
                pin_memory = True)

"""
train_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.train_list, type_of_data = "train",transform = None),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = args.workers, 
                pin_memory = True)

val_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.val_list,type_of_data = "val",transform = None),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = args.workers, 
                pin_memory = True)


# validate before starting to train
validate(val_loader, model, criterion)


# train and validate    
for epoch in range(args.start_epoch, args.epochs):
#    scheduler.step()
    
    # train for one epoch
    train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args.step)

    # evaluate on validation set
    val_loss = validate(val_loader, model, criterion)
    

    save_name = args.save_path + 'args.model_' + 'args.dataset_' +str(epoch+1) + '_checkpoint.pth.tar'
    save_checkpoint({
        'epoch' : epoch + 1,
        'model' : args.model,
        'datset' : args.dataset,
        'state_dict': model.state_dict(),
        'train_loss' : train_loss,
        'val_loss' : val_loss},
        #'test_loss' : test_loss}, 
         save_name)     
    


