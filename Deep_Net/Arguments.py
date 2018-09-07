import os
from argparse import ArgumentParser

#lr = 0.01/(320*240)

parser = ArgumentParser(description = 'Saliency Detection')    
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-step', default = 2000, type = int, metavar = 'N',
                    help='number of iterations to halve LR')
parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default = 5, type = int, metavar = 'N',
                    help = 'number of total epochs to run')
parser.add_argument('--start_epoch', default = 0, type = int, metavar = 'N',
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default = 2, type = int,
                    metavar = 'N', help = 'mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default = 1.30208333333e-07 , type = float,
                    metavar = 'LR', help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                    help = 'momentum')
parser.add_argument('--weight_decay', '--wd', default = 5e-07, type = float,
                    metavar = 'W', help = 'weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default = 100, type = int,
                    metavar='N', help='print frequency (default: 100)')    
parser.add_argument('--resume', default = '', type = str, metavar = 'PATH',
                    help = 'path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default = './SaliconDataset/', type = str, metavar = 'PATH',
                    help = 'path to root path of images (default: none)')
parser.add_argument('--train_list', default = './SaliconDataset/trainList.txt', type = str, 
metavar = 'PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default = './SaliconDataset/valList.txt', type = str, metavar = 
'PATH',
                    help = 'path to validation list (default: none)')
parser.add_argument('--test_list', default = './SaliconDataset/testList.txt', type = str, metavar 
= 'PATH',
                    help = 'path to validation list (default: none)')
parser.add_argument('--save_path', default = './Deep_Net/Model_Save_new/', type = str, metavar = 'PATH',
                    help = 'path to save checkpoint (default: none)')
parser.add_argument('--model', default = 'DeepNet', type = str, metavar = 'Model',
                    help = 'model type: DeepNet')
parser.add_argument('--dataset', default = 'SALICON ', type = str, metavar = 'Dataset',
                    help = 'Dataset: SALICON')
    
def get_args():
    args = parser.parse_args('--cuda True'.split())    
    return args

