import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
import numpy as np
import argparse
from DeepNet_model import DeepNet
from scipy import ndimage
from scipy import misc
#added arguments.py
from Arguments import get_args
from PIL import Image, ImageOps
from load_imglist import ImageList


global args #defining variables that can be accessed globally
global device #defining variables that can be accessed globally
global count
count = 1
#args = parser.parse_args()
args = get_args()
device = torch.device("cuda" if args.cuda else "cpu")

def test(test_loader, model):
    model.eval()
    print('entered test')
    for i, (ip, gt) in enumerate(test_loader):
        ip = ip.to(device)
        predict(model, ip, i)

def predict(model, image_stimuli,img_no):
    global count
    print('predicted image no {}'.format(img_no))
    predicted = model(image_stimuli)
    for i in range(predicted.shape[0]):
        img= predicted[i]
        #img = img.detach().cpu().numpy().reshape(176,176)
        img = img.detach().cpu().reshape(176,176)
        #img = img*128
        #mean = torch.FloatTensor([27])
        #img = img + mean
        img = ndimage.gaussian_filter(img,sigma=2)
        img = misc.imresize(img,(180,180))#/255
        #img = img/img.max()
        #img = Image.fromarray(np.uint8(img*255),'L')
        img = Image.fromarray(np.uint8(img),'L')
        #img = ImageOps.invert(img)
        img.save("/home/iimtech5/Deep_Net/result/"+ "out_{:04d}.png".format(count))
        count +=1

test_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.test_list,transform = None),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = args.workers, 
                pin_memory = True)



model = DeepNet()
model.load_state_dict(torch.load('./Model_Save_new/args.model_args.dataset_5_checkpoint.pth.tar')['state_dict'])
model = model.to(device)
print(model)
print('model loaded and test started')
#model.eval()
test(test_loader, model)
