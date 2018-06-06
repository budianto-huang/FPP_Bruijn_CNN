import scipy.signal
import numpy as np
import tempfile
import uuid
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from subprocess import call


from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from numpy.linalg import inv


def CalcPhi(phi):
    sz = phi.shape
    
    print(phi.shape)
    y = []
    if len(sz) == 2: 
        y = scipy.signal.hilbert(phi, axis=0)
    
    
    return np.angle(y)

def CalcPhiComplex(rec):
    I1 = rec[:,:,0]
    I2 = rec[:,:,1]
    I3 = rec[:,:,2]    
    rl = np.sqrt(3)*(I1-I3)
    rl = rl.reshape((1, rl.shape[0], rl.shape[1]))
    im = (2*I2-I1-I3)
    im = im.reshape((1, im.shape[0], im.shape[1]))
#         angle = np.arctan2(rl,im)
#         mag = np.sqrt() 
    result = np.concatenate((rl,im), axis=0)

    return result

def CalcPhi3(rec):
    I1 = rec[:,:,0]
    I2 = rec[:,:,1]
    I3 = rec[:,:,2]    
    rl = np.sqrt(3)*(I1-I3)
    im = (2*I2-I1-I3)
    angle = np.arctan2(rl,im)
#         mag = np.sqrt()        
    return angle

def unwrapGold(phase, mask): 
    
    pre = tempfile.gettempdir() +"/" + str(uuid.uuid4()) + "_"
        
    phaseFilename = pre + 'im.phase'
    maskFilename = pre + 'im.mask'
    outFilename = pre + 'im.out'    
           
    phase = phase + np.pi
    
    # write the phse information to file
    f = open(phaseFilename, "w")    
    phase.astype(dtype=np.float32).tofile(f)
    f.close()
    
    # write the mask information to file
    f = open(maskFilename, "wb+")
    mask = mask.astype(dtype=np.uint8)    
    mask.tofile(f)
    f.close()
    
    run_goldstein(phaseFilename, maskFilename, outFilename, phase.shape)
    
    
    f = open(outFilename, "r+")
    result = np.fromfile(f, dtype=np.float32)    
    result = np.reshape(result, phase.shape)
    f.close()
    
    return result

def run_goldstein(phaseFilename, maskFilename, outFilename, sz, debug = 'no'):
    cmd = 'extras/goldstein' 
    cmd = cmd + ' -input ' + phaseFilename 
    cmd = cmd + ' -output ' + outFilename    
    cmd = cmd + ' -mask ' + maskFilename 
    cmd = cmd + ' -xsize {} -ysize {} '.format(sz[0], sz[1])    
    cmd = cmd + ' -dipole no -format float '    
    cmd = cmd + ' -debug ' + debug + ' ' 
    
    print(cmd)       
    call(cmd, shell=True)

def SNR(GT, Result, mask):
    MSE = np.multiply(GT-Result, mask)
    MSE_Ori = np.multiply(GT, mask)
    #s = np.std(MSE)
    
    total = np.sum(mask)
    
    
    MSE = np.sum(MSE**2) / total
    
    MSE_Ori = np.sum(MSE_Ori**2) / total
    
    return 20.0 * np.log10(MSE_Ori/MSE) 
    
def imagesc(I, mask = 0, dpi_=70, colorbar_=True, cmap_='jet', mn=0, mx=0, title_='untitled'):
    if type(mask) is int and mask == 0:
        mask = np.ones((I.shape))
#         print("mask size = ", mask)
    fig = figure(0,dpi=dpi_, facecolor='w', edgecolor='k')
    if title_ != 'untitled':
        fig.suptitle(title_)
    plt.imshow(I * mask, cmap=cmap_)
    if colorbar_:
        plt.colorbar()
    if mx > mn:
        plt.clim(mn, mx)
    plt.show()  
    
def plot(I, dpi_=70, strFormat='b'):
    figure(0,dpi=dpi_, facecolor='w', edgecolor='k')    
    plt.plot(I, strFormat)
          

def imshow(I, dpi_=70, colorbar_=False, mn=0, mx=255):
    
    figure(1,dpi=dpi_, facecolor='w', edgecolor='k')
    if len(I.shape) < 3:
        plt.imshow(I, cmap='gray')
    else:    
        plt.imshow(I)
    if colorbar_:
        plt.colorbar()
    if mx > mn:
        plt.clim(mn, mx)
    plt.show()      
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def CalcSNR_MSE(I, IOri, mask=0):
    if len(mask) == 1:
        mask = np.ones(I.shape)
    mseShape = ((I - IOri) * mask) ** 2
    mseShape = mseShape * (mseShape < 10)
    mse = np.sum(mseShape)/(np.sum(mask))
    
    ori = np.sum(np.multiply(IOri,mask) ** 2)/np.sum(mask)
    
    SNR = 10*np.log10(ori/mse)
    print("SNR = ", SNR)
    
    return SNR, mse, mseShape 



def IRLS(y, X, maxiter, w_init = 1, d = 0.0001, tolerance = 0.001):
    n,p = X.shape
    delta = array( repeat(d, n) ).reshape(1,n)
    w = repeat(1, n)
    W = diag( w )
    B = dot( inv( X.T.dot(W).dot(X) ), 
             ( X.T.dot(W).dot(y) ) )
    for _ in range(maxiter):
        _B = B
        _w = abs(y - X.dot(B)).T
        w = float(1)/maximum( delta, _w )
        W = diag( w[0] )
        B = dot( inv( X.T.dot(W).dot(X) ), 
                 ( X.T.dot(W).dot(y) ) )
        tol = sum( abs( B - _B ) ) 
        
#         print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    return B
