
import matplotlib.image as mpimg
import numpy as np

def ReadFringeImages(prefixFld, imgFld, prefixImg, N, offset, sz):
    
    filename = prefixFld + '{0}/{1}{2}.jpg'.format(imgFld, prefixImg, 1)
    print(prefixFld)
    print('{0}/{1}{2}.jpg'.format(imgFld, prefixImg, 1))
    
    
    tmp = mpimg.imread(filename)
    fringe = tmp[offset[0]:offset[0]+sz,offset[1]:offset[1]+sz,:]
    
    
    ref_A = np.zeros((sz, sz, 3))
    ref_B = np.zeros((sz, sz, 3))
    flat  = np.zeros((sz, sz, 3))
    for i in range(N):
        filename = prefixFld + '{0}/{1}{2}.jpg'.format(imgFld, prefixImg, i+1)
        tmp = mpimg.imread(filename).astype(np.float)
        tmp = tmp[offset[0]:offset[0]+sz,offset[1]:offset[1]+sz,:]
        if i == 0:
            ref_A[:, :, 0] = tmp[:, :, 0] * np.sin(2.0*np.pi*i/N - 2*np.pi/3)
            ref_B[:, :, 0] = tmp[:, :, 0] * np.cos(2.0*np.pi*i/N - 2*np.pi/3)
            ref_A[:, :, 1] = tmp[:, :, 1] * np.sin(2.0*np.pi*i/N)
            ref_B[:, :, 1] = tmp[:, :, 1] * np.cos(2.0*np.pi*i/N)
            ref_A[:, :, 2] = tmp[:, :, 2] * np.sin(2.0*np.pi*i/N + 2*np.pi/3)
            ref_B[:, :, 2] = tmp[:, :, 2] * np.cos(2.0*np.pi*i/N + 2*np.pi/3)
            flat = tmp
        else:
            ref_A[:, :, 0] = tmp[:, :, 0] * np.sin(2.0*np.pi*i/N - 2*np.pi/3) + ref_A[:, :, 0]
            ref_B[:, :, 0] = tmp[:, :, 0] * np.cos(2.0*np.pi*i/N - 2*np.pi/3) + ref_B[:, :, 0]
            ref_A[:, :, 1] = tmp[:, :, 1] * np.sin(2.0*np.pi*i/N) + ref_A[:, :, 1]
            ref_B[:, :, 1] = tmp[:, :, 1] * np.cos(2.0*np.pi*i/N) + ref_B[:, :, 1]
            ref_A[:, :, 2] = tmp[:, :, 2] * np.sin(2.0*np.pi*i/N + 2*np.pi/3) + ref_A[:, :, 2]
            ref_B[:, :, 2] = tmp[:, :, 2] * np.cos(2.0*np.pi*i/N + 2*np.pi/3) + ref_B[:, :, 2]
            flat = tmp + flat; 

    
    ref_A = np.sum(ref_A, axis=2) / 3
    ref_B = np.sum(ref_B, axis=2) / 3
    phi_GT = np.arctan2(ref_A, ref_B)
    flat = flat / N 
    
    return fringe, phi_GT, flat

def ReadImage(prefixFld, imgFld, prefixImg, offset, sz):
    
    filename = prefixFld + '{0}/{1}.jpg'.format(imgFld, prefixImg)
    print(filename)
    I = mpimg.imread(filename)    
    return I[offset[0]:offset[0]+sz,offset[1]:offset[1]+sz,:]   