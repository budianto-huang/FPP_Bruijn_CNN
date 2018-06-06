import matplotlib.image as mpimg
import numpy as np
import os.path
import cv2
import scipy.io as sio


class ImageLoader():
    def __init__(self, dir):
        dir_ = dir
        # load the GROUND TRUTH DATA
        #  {'GT_mask' : mask, 'GT_flat' : flat, 'GT_Fringe': zeroFringe, 'GT_rRoi': rowR, 'GT_cRoi': colR}
        tmp = sio.loadmat(dir_ + '/' + '{0}.mat'.format('GT'))
        self.flat = tmp['GT_flat']
        self.mask = tmp['GT_mask']
        self.zeroFringe = tmp['GT_Fringe']
        self.rRoi = tmp['GT_rRoi'][0]
        self.cRoi = tmp['GT_cRoi'][0]

        ftype = 'FDB'
        self.colorFringe = cv2.imread(dir_ + '/' + '{0}_{1}.jpg'.format(ftype, 1))

        self.cropAll()

    # def loadAnImage(filename):
    #     # Load the Flat image
    #     return cv2.imread(filename)  # Read in color image from filename

    def cropAll(self):
        rRoi = self.rRoi
        cRoi = self.cRoi
        print("Crop ", rRoi[0], cRoi)
        print(self.zeroFringe.shape)
        self.zeroFringe = self.zeroFringe[rRoi[0]:rRoi[1], cRoi[0]:cRoi[1]]
        self.colorFringe = self.colorFringe[rRoi[0]:rRoi[1], cRoi[0]:cRoi[1], :]

    def getFlat(self):
        return self.flat

    def getFringe(self):
        # return self.fr
        return self.colorFringe

    def getFringeZero(self):
        return self.zeroFringe
