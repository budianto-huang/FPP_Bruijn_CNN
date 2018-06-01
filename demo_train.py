# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:08:54 2017

@author: buddy
"""

import argparse
import os
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from models.model_wav import _netG
from models.model_wav import *
from DnnFPP.BSD import BSD 
import cv2

from tensorboardX import SummaryWriter

import numpy as np



parser = argparse.ArgumentParser(description="Training for Bias Removal")

parser.add_argument('--epochs',                 default=5,      type=int,   metavar='N',    help='manual epoch number')
parser.add_argument('--lr', '--learning-rate',  default=0.01,   type=float, metavar='LR',   help='initial learning rate')
parser.add_argument('-b', '--batch-size',       default=128,    type=int,   metavar='N',    help='mini-batch size(default: 128')
parser.add_argument('--patch-size',             default=40,     type=int,   metavar='N',    help='Patch size for training (default: 40x40')
parser.add_argument('--resume',                 default='',     type=str,   metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch',            default=0,      type=int,   metavar='N',    help='manual epoch number (useful on restarts)')
parser.add_argument('--weight-decay', '--wd',   default=1e-4,   type=float, metavar='W',    help='weight decay (default: 1e-4)')
parser.add_argument('--WLevel',                 default=4,      type=int,   metavar='N',    help='Level of Wavelet')
parser.add_argument('--model',                  default='B',     type=str,   metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--losstype',              default='1',     type=str,   metavar='PATH', help='path to latest checkpoint (default: none)')




parser.add_argument('-e', '--evaluate', dest='evaluate',    action='store_true',    help='evaluate model on validation set')
# parser.add_argument('--pretrained',     dest='pretrained',  action='store_true',    help='use pre-trained model')

def main():    
    global args
    args = parser.parse_args()
    
    outChannel = 1
    inChannel = 3
    
#     netG = resnet18b() #_netG(in_channel = nChannel, out_channels = nChannel, init_weights=True)
    currModelType = args.model    
    netG = _netG(in_channel = inChannel, out_channels = outChannel, init_weights=True, modelType=currModelType)
    netG.cuda()
    
    
    # Loss function
    if args.losstype == '1':
        criterion = nn.MSELoss()
    elif  args.losstype == '2': # L1 Loss
        criterion = nn.L1Loss()
    elif  args.losstype == '3': # L1 Loss
        criterion = nn.SmoothL1Loss()
    elif args.losstype == '4': # KLDiv
        criterion = nn.KLDivLoss()
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif args.losstype == '5':  # L1 Loss
        criterion = nn.PoissonNLLLoss(False)


    #criterion = nn.SmoothL1Loss()
    #criterion2 = nn.SmoothL1Loss()
    
    
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, dampening=0, weight_decay=0.0001)
    optimizer = optim.Adam(netG.parameters(), lr=args.lr)
    

    #gma = 0.8709635899560807 # math.exp(math.log(1e-3)/args.epochs)
    gma = 0.5
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=gma)
    
    
    if args.evaluate and args.resume:   
        
        return
    
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            netG.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Loading DATA 
    #batch_size = 128
    #patchSz = 40
    transform = transforms.Compose(
        [transforms.ToTensor()])
        #,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    patchInc = 8





    currPatchSize = args.patch_size
    patchInc = 8
    writername = 'runs/%s/train_30_05_18_sz=%d_LR=%.4f_loss=%s' % (args.model, args.patch_size, args.lr, args.losstype)
    writer = SummaryWriter(writername, comment='kmap_training')
    # Train the Model
    isFirstTime = True
    for epoch in range(args.start_epoch, args.epochs):

        if epoch % 10 == 0 or isFirstTime:
            currPatchSize = args.patch_size + round(epoch / 10) * patchInc  # * 2**(round(epoch / 10))
            isFirstTime = False
            print("=====================  Current PatchSize=", currPatchSize)
            evalset = BSD(root='', patchSz=currPatchSize, train=False, transform=transform, generate_dataset=False,  WLevel=args.WLevel)
            evalloader = torch.utils.data.DataLoader(evalset, args.batch_size, shuffle=True)

            trainset = BSD(root='', patchSz=currPatchSize, train=True, transform=transform, generate_dataset=False, WLevel=args.WLevel)
            trainloader = torch.utils.data.DataLoader(trainset, args.batch_size, shuffle=True)


        exp_lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        
        train(trainloader, netG, criterion, optimizer, epoch, writer)
        eval(evalloader, netG, criterion, epoch, writer)

        prefix = 'cp'
        fname = '{0}_{1}_{2}_{3}_{4}.pth'.format(prefix, args.patch_size, args.lr, epoch, args.losstype)
        
#         if (epoch + 1) % 1 == 0:
        save_checkpoint({        
            'epoch'     : epoch + 1,
            'state_dict': netG.state_dict(),
            'optimizer' : optimizer.state_dict()
            },
            filename = fname,
            modelType = currModelType
        )
    
    writer.close()
    print('Finished Training')
    
    #         if i == 0:
    #             imshowgrid(torchvision.utils.make_grid(images))
    #             imshowgrid(torchvision.utils.make_grid(labels))
    #             type(images)

    
def train(trainloader, netG, criterion, optimizer, epoch, writer):
    
    
    netG.train()
    
    embedding_log = 5
    totalLoss = 0
    for i, (images, labels) in enumerate(trainloader):
        n_iter = (epoch*len(trainloader)) + i
        #print('image size = ', images.shape)
        
        I = images.numpy()
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        out = netG(images)
        
        loss = criterion(out, labels)
        
        
        # BACKWARD
        loss.backward()
        optimizer.step()
        totalLoss += loss.data[0]
        if (i + 1) % 500 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, args.epochs, i + 1, len(trainloader), loss.data[0]))
            #print("type ===========", type(out.data))            
#             out = torch.cat((out.cpu().data, torch.ones(len(out), 1)), 1)
#             writer.add_embedding(out, metadata=labels.data, label_img=images.data, global_step=n_iter)
        
    
    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   % (epoch + 1, args.epochs, i + 1, len(trainloader), totalLoss))
    
    writer.add_scalar('loss', totalLoss, epoch)

def eval(trainloader, netG, criterion, epoch, writer):
    
    
    netG.eval()
    print("Length ====== ", len(trainloader))
    corrects, avg_loss = 0, 0.0
    i = 0
    total = 0
    
    for batch in trainloader:
        n_iter = (epoch*len(trainloader)) + i
        #print('image size = ', images.shape)
        (images, labels) = batch
        
        I = images.numpy()
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        
        out = netG(images)        
        loss = criterion(out, labels)
        avg_loss += loss.data[0].double()

        corrects += (torch.round(out).data == labels.data).sum() 
        total += (labels.data>0).sum()
    
    
    #size = len(trainloader.dataset) * args.patch_size * args.patch_size
    size = total.double()
    avg_loss /= size.double()
    accuracy = 100.0 * corrects.double()/size
    

    
    writer.add_scalar('loss_eval', avg_loss.data[0], epoch)
    writer.add_scalar('accuracy', accuracy.data[0], epoch)

def save_checkpoint(state, filename='cp_.pth', modelType = 'B2'):    
    filename = 'saved/' + modelType + '/' + filename
    torch.save(state, filename)
    print("=> check point is saved at '{}'".format(filename))
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')    
    
if __name__ == '__main__':
    main()    
