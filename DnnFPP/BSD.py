# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:52:57 2017

@author: buddy
"""

# import pdb; pdb.set_trace()
from __future__ import print_function
from sklearn.feature_extraction import image
import os.path
import numpy as np
from imgaug import augmenters as aug
import torch.utils.data as data
from DnnFPP.ImageLoader import * 
import scipy.io as sio
from torchvision import transforms
#from sklearn.mixture.tests.test_gaussian_mixture import generate_data



class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
        N (int): Number of cropped samples
    """
    def __init__(self, output_size, N):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.N = N

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        print("Image shape = ", image.shape)
        print("label shape = ", label.shape)

        h, w = image.shape[:2]
        Nc = 1
        if len(image.shape) == 3:
            Nc = 3
        new_h, new_w = self.output_size
        sampN = self.N

        top = np.random.randint(0, h - new_h, size=sampN)
        left = np.random.randint(0, w - new_w, size=sampN)
        #print("top = ", top)

        image = image.astype(np.float32)
        imgList = np.zeros((sampN, new_h, new_w, Nc))
        lblList = np.zeros((sampN, new_h, new_w))
        for r in range(new_h):
            for c in range(new_w):
                if Nc == 3:
                    imgList[:, r, c, 0] = image[top + r, left + c, 0]  # image[top: top + new_h, left: left + new_w]
                    imgList[:, r, c, 1] = image[top + r, left + c, 1]  # image[top: top + new_h, left: left + new_w]
                    imgList[:, r, c, 2] = image[top + r, left + c, 2]  # image[top: top + new_h, left: left + new_w]
                else:
                    imgList[:, r, c, 0] = image[top+r, left+c]  #image[top: top + new_h, left: left + new_w]

                lblList[:, r, c] = label[top+r, left+c]  # image[top: top + new_h, left: left + new_w]


        return {'imageList': imgList, 'labelList': lblList}

class BiasJitter(object):
    def __init__(self, biasLevel):
        assert len(biasLevel) == 2
        self.biasLevel = biasLevel
    def __call__(self, sample):
        imgList, lblList  = sample['imageList'], sample['labelList']
        M, N, NSamp  = imgList.shape

        bias = np.random.randint(self.biasLevel[0], self.biasLevel[1], size=NSamp)
        bias = bias.reshape((1, 1, NSamp))
        bias = np.repeat(np.repeat(bias, M, axis=0),N,axis=1)

        augImgList = imgList + bias

        imgList = np.concatenate((imgList, augImgList), axis=2)
        lblList = np.concatenate((lblList, lblList), axis=2)
        return {'imageList': imgList, 'labelList': lblList}


class RandomRotate(object):
    def __init__(self, degree):
        assert isinstance(degree, (int, tuple))
        if isinstance(degree, int):
            self.degree = (-degree, degree)
            print(self.degree)
        else:
            assert len(degree) == 2
            self.degree = degree

        self.seq = aug.Affine(rotate=(self.degree[0], self.degree[1]))

    def __call__(self, sample):
        imgList, lblList = sample['imageList'], sample['labelList']
        I = np.concatenate((imgList, lblList), axis=3)
        print("list shape = ", I.shape)
        print(type(I[0, 0, 0, 0]))
        seq_det = self.seq.to_deterministic()
        IAug = seq_det.augment_images(I.astype(np.uint8))

        print(IAug[:, :, :, 0].shape)
        print(imgList.shape)
        #         imgList = np.concatenate((imgList, IAug[:,:,:,0]), axis=0);
        #         lblList = np.concatenate((lblList, IAug[:,:,:,1]), axis=0);

        #         imgList = imgList.transpose((3,1,2,0))
        #         lblList = lblList.transpose((3,1,2,0))

        return {'imageList': imgList, 'labelList': lblList}

class To4D(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imgList, lblList = sample['imageList'], sample['labelList']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        NSamp, M, N, Nc = imgList.shape

        # imgList = imgList.reshape((1, M, N)).astype(np.float32)
        lblList = lblList.reshape((NSamp, M, N, 1)).astype(np.float32)
        imgList = imgList.transpose((0, 3, 1, 2)).astype(np.float32)
        lblList = lblList.transpose((0, 3, 1, 2)).astype(np.float32)

        return {'imageList': imgList, 'labelList': lblList}

class BSD(data.Dataset):
    """ Get the data """

    def __init__(self, root, patchSz=40, train=True, transform=None, target_transform=None, generate_dataset=False, WLevel=5):
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.patchSz = patchSz


        self.input = []
        self.output = []
        if train==False: 
            self.genDatasetEval()
            print("Eval")
        else:
            self.genDataset2()
            print("Training data")    
            
            
        # now load the picked numpy arrays
        if train == True:
            self.train_data = []
            self.train_labels = []            
            
            samplingN = 2000
            maxN = 0
            print("==== total number of sample file = ", len(self.input))
            for i in range(len(self.input)):
                transform = transforms.Compose([RandomCrop(self.patchSz, samplingN),
                                                #BiasJitter([-20, 20]),
                                                To4D()])
                tmpData = transform({'image': self.input[i], 'label': self.output[i]})

                if i == 0:
                    self.train_data = tmpData['imageList']#np.concatenate((trainData, augTrain), axis=0)
                    self.train_labels = tmpData['labelList']#np.concatenate((trainLabels, trainLabels), axis=0)
                else:
                    self.train_data = np.concatenate((self.train_data, tmpData['imageList']), axis=0)
                    self.train_labels = np.concatenate((self.train_labels, tmpData['labelList']), axis=0)


                print("Extraing set-", i ," and extracted size = ", tmpData['imageList'].shape, " total = ", self.train_data.shape)
            print("Final Total shape =",self.train_data.shape)
        elif train == False:
            self.train_data = []
            self.train_labels = []
            maxN = 60000
            transform = transforms.Compose([RandomCrop(self.patchSz, maxN),
                                            #BiasJitter([-20, 20]),
                                            To4D()])

            tmpData = transform({'image': self.input[0], 'label': self.output[0]})
            self.train_data = tmpData['imageList']  # np.concatenate((trainData, augTrain), axis=0)
            self.train_labels = tmpData['labelList']  # np.concatenate((trainLabels, trainLabels), axis=0)
            print("Evaluation data shape=",self.train_data.shape)
        print('Data loading finished')

    def genDataset2(self):
        
        #tmp = sio.loadmat('KMapData/flat_KMap2.mat')
        
        
        # REF 1
        ILoad = ImageLoader('../CapImages/Set6/' + 'ref1')
        self.input.append(ILoad.getFringe().astype(np.float32))    
        self.output.append(ILoad.getFringeZero().astype(np.float32))

        ILoad = ImageLoader('../CapImages/Set6/' + 'ref2')
        self.input.append(ILoad.getFringe().astype(np.float32))
        self.output.append(ILoad.getFringeZero().astype(np.float32))

        ILoad = ImageLoader('../CapImages/Set6/' + 'ref3')
        self.input.append(ILoad.getFringe().astype(np.float32))
        self.output.append(ILoad.getFringeZero().astype(np.float32))

        # ILoad = ImageLoader('../CapImages/Set6/' + 'ref4')
        # self.input.append(ILoad.getFringe().astype(np.float32))
        # self.output.append(ILoad.getFringeZero().astype(np.float32))


       # # HEAD 1
       #  #ILoad = ImageLoader('../CapImages/Set4/' + 'head1')
       #  ILoad = ImageLoader('../CapImages/Set5/' + 'head1')
       #  flatRec = ILoad.getFlat().astype(np.float32)
       #  maskRec = (flatRec - flatRef) > 10
       #  Mm, Nm = maskRec.shape
       #  maskRec3 = maskRec.reshape((Mm, Nm,1))
       #  rRoi, cRoi = maskRec.nonzero()
       #
       #  tmp = ILoad.getFringe().astype(np.float32) * np.repeat(maskRec3, 3, axis=2)
       #  # print(tmp.shape, rRoi, cRoi)
       #  tmp = tmp[np.min(rRoi):np.max(rRoi),np.min(cRoi):np.max(cRoi)]
       #  self.input.append(tmp)
       #  tmp = ILoad.getFringeZero().astype(np.float32) * maskRec
       #  tmp = tmp[np.min(rRoi):np.max(rRoi), np.min(cRoi):np.max(cRoi)]
       #  self.output.append(tmp)
       #
       #  # CAT
       #  #ILoad = ImageLoader('../CapImages/Set4/' + 'cat')
       #  ILoad = ImageLoader('../CapImages/Set5/' + 'cat2')
       #  flatRec = ILoad.getFlat().astype(np.float32)
       #  maskRec = (flatRec-flatRef) > 3
       #  Mm, Nm = maskRec.shape
       #  maskRec3 = maskRec.reshape((Mm, Nm, 1))
       #  rRoi, cRoi = maskRec.nonzero()
       #
       #  tmp = ILoad.getFringe().astype(np.float32) * np.repeat(maskRec3, 3, axis=2)
       #  tmp = tmp[np.min(rRoi):np.max(rRoi), np.min(cRoi):np.max(cRoi)]
       #  self.input.append(tmp)
       #  tmp = ILoad.getFringeZero().astype(np.float32) * maskRec
       #  tmp = tmp[np.min(rRoi):np.max(rRoi), np.min(cRoi):np.max(cRoi)]
       #  self.output.append(tmp)
        
        print('Generate Dataset finished')
    
    def genDatasetEval(self):

        ILoad = ImageLoader('../CapImages/Set6/' + 'ref4')
        self.input.append(ILoad.getFringe().astype(np.float32))
        self.output.append(ILoad.getFringeZero().astype(np.float32))
        
        print('Generate Evaluation Dataset finished')       
    
    
    def __len__(self):
        return (self.train_data.shape[0])

    def __getitem__(self, idx):
        img, target = self.train_data[idx], self.train_labels[idx]
        return img, target

    
    
        
    def flist_reader(self, txtFile):
        """
        flist format: impath label\nimpath label\n ...(same to caffe's filelist)
        """
        root = self.root
        fpath = os.path.join(root, self.base_folder, txtFile)
        imlist = []
        with open(fpath, 'r') as rf:
            for line in rf.readlines():
                impath = line.strip()
                imlist.append(impath)

        return imlist
    
    def normalize(self, I):
        mn = np.min(I)
        mx = np.max(I)


        I = ((I - mn) / (mx-mn))
#         I = I.astype(np.uint8)
        return I, mn, mx

