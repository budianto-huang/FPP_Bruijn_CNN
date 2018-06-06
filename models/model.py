# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:46:34 2017

    net = VGG2('VGG16B')

@author: buddy
"""

'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn
import math
# from models.median_pool import MedianPool2d



class _netG(nn.Module):
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def __init__(self, kernelSz=3, in_channel = 3, out_channels = 1, init_weights=True, num_classes=6, modelType='B2'):        
        super(_netG, self).__init__()        
        print("Constructor")

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1, bias=True)
        self.modelType = modelType
        self.layers,LastChannels = make_layers(cfg[modelType], in_channels=in_channel, kernelSz=7, batch_norm=True)
        self.convN = nn.Conv2d(LastChannels, out_channels, kernel_size=3, padding=1, bias=True)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        print("length of layers = ", len(self.layers))
        self.layers2 = nn.Sequential(*self.layers)
        print("length of layers = ", len(self.layers2))

    def forward(self, x):

        for i in range(len(self.layers)):
            x = self.layers[i](x)
        # x = self.layers2(x)
        x = self.convN(x)

        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                

    
def make_layers(cfg, in_channels=3, kernelSz=3, batch_norm=True):
    layers = []
    first_time = True    
    for v in cfg:        
        if v == 'D':                
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            #layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers  += [nn.Upsample(scale_factor=2, mode='bilinear')]
        # elif v == 'M':
        #     convMed = MedianPool2d(kernel_size=3, stride=1, padding=1, same=True)
        #     layers  += [convMed]
        elif v == 'DR':
            dropout = nn.Dropout2d(p=0.5)
            layers += [dropout]
        else:
            paddingSz = int((kernelSz-1)/2)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=kernelSz, padding=paddingSz)
            if first_time or not batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.BatchNorm2d(v),nn.Softshrink()]
                first_time = False
            else:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.BatchNorm2d(v), nn.Softshrink()]
            in_channels = v
    return layers, in_channels



cfg = {
    'A': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], # ==> quite good

    'B': [32, 32, 32, 32, 32], # ==> quite good

    'C': [64, 64, 'D', 128, 128, 'D', 256, 256, 'U', 128, 128, 'U', 64, 64], # ==> quite good

    'D': [64, 64, 'D', 32, 32, 'D', 16, 16, 'U', 8, 8, 'U', 4, 4], # ==> quite good
    
    'E': [64, 64, 64, 'D', 32, 32, 32, 'D', 16, 16, 16, 'D', 8, 8, 8, 'U', 16, 16, 16, 'U', 32, 32, 32, 'U', 64, 64, 64], # ==> quite good

    'F': [64, 'D', 128, 128, 'U', 64, 64, 'D', 128, 128, 'U', 64, 64, 'D', 128, 128, 'U', 64, 64], # ==> quite good

    'G': [64, 64, 64, 64, 64, 64, 'D', 32, 32, 32, 32, 32, 32,  'U', 64, 64, 64, 64, 64, 64], # ==> quite good

    'H': [64, 64, 64, 64, 'D', 32, 32, 32, 32, 'D', 16, 16, 16, 16, 16, 16, 'U', 32, 32, 32, 32, 'U', 64, 64, 64, 64], # ==> quite good

    'I':   [64, 64, 64, 64, 64, 64, 'D', 32, 32, 32, 32, 32, 32, 'D', 16, 16, 16, 16, 16, 16, 16, 16, 'U', 32, 32, 32, 32, 32, 32, 'U', 64, 64, 64, 64, 64, 64], # ==> quite good

}

