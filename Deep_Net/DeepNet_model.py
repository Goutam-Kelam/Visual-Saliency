import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init								

class DeepNet(nn.Module):
    def __init__(self, dataset = 'SALICON'):
        super(DeepNet,self).__init__()
        self.dataset = dataset
        #loading pretrained weights in dictionary 'd'
        #self.d = torch.load('./vggm-786f2434.pth')

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 7,stride=1,padding=3)
        self.act1 = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(5) #used in implementation
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride=1, padding =2)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride=1, padding =1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 5, stride=1, padding =2)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 5, stride=1, padding =2)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 7, stride=1, padding =3)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 11, stride=1, padding =5)
        self.act7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 11, stride=1, padding =5)
        self.act8 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 13, stride=1, padding =6)
        #self.act9 = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 8, stride=4, padding =2, bias =False)
        self.initWeights()
    


    """def initWeights(self):
        self.conv1.weight = nn.Parameter(self.d['features.0.weight'])
        self.conv1.bias = nn.Parameter(self.d['features.0.bias'])
        self.conv2.weight = nn.Parameter(self.d['features.4.weight'])
        self.conv2.bias = nn.Parameter(self.d['features.4.bias'])
        self.conv3.weight = nn.Parameter(self.d['features.8.weight'])
        self.conv3.bias = nn.Parameter(self.d['features.8.bias'])
        
        init.normal_(self.conv4.weight)
        init.kaiming_uniform_(self.conv5.weight)
        init.kaiming_uniform_(self.conv6.weight)
        init.kaiming_uniform_(self.conv7.weight)
        init.kaiming_uniform_(self.conv8.weight)
        init.kaiming_uniform_(self.conv9.weight)
        init.kaiming_uniform_(self.deconv1.weight)
     """ 
    def initWeights(self):
        init.normal_(self.conv1.weight,0,0.116642)
        init.constant_(self.conv1.bias,0)
        init.normal_(self.conv2.weight,0,0.028867)
        init.constant_(self.conv2.bias,0)
        init.normal_(self.conv3.weight,0,0.029462)
        init.constant_(self.conv3.bias,0)
        init.normal_(self.conv4.weight,0,0.0125)
        init.constant_(self.conv4.bias,0)
        init.normal_(self.conv5.weight,0,0.0125)
        init.constant_(self.conv5.bias,0)
        init.normal_(self.conv6.weight,0,0.008928)
        init.constant_(self.conv6.bias,0)
        init.normal_(self.conv7.weight,0,0.008035)
        init.constant_(self.conv7.bias,0)
        init.normal_(self.conv8.weight,0,0.011363)
        init.constant_(self.conv8.bias,0)
        init.normal_(self.conv9.weight,0,0.013598)
        init.constant_(self.conv9.bias,0)
        init.normal_(self.deconv1.weight,0.015625,0.000001)
                 


    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.lrn(x)
        x = self.maxpool1(x)
        x = self.maxpool2(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.act8(self.conv8(x))
        x = self.conv9(x)
        x = self.deconv1(x)
        return x
    
    
