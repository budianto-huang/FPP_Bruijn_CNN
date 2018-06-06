import numpy as np
import dtcwt
from dtcwt.numpy import Transform2d
from scipy.signal import medfilt2d


def DTCWT_Forw(G, J):
    DTCWT = Transform2d()
    return DTCWT.forward(G, nlevels=J)

def DTCWT_Inv(w):
    DTCWT = Transform2d()
    return DTCWT.inverse(w)
    
def dtcwt_denoiseFPP(Y, MaxIter = 0):
    J = 5
    G = Y
    g_w = DTCWT_Forw(G, J)
    
    print(g_w.lowpass.shape)
    # Set zero for the scaleing 
    #for k in range(len(g_w.lowpass)):
    g_w.lowpass = np.multiply(g_w.lowpass[:, :], 0)
    print(g_w.lowpass.shape)
     
    tmp  = g_w.highpasses[1][:, :, 1]
    #Th  = np.median(abs(tmp))/0.6745
    Th = np.max(abs(tmp)) / 0.6745
    
    g_w = denoiseW(g_w, Th)
    
    
    
    Y_est = DTCWT_Inv(g_w)
    
    
    for k in range(MaxIter):
        tmp =  DTCWT_Forw(Y_est - DTCWT_Inv(g_w), J) 
        J = len(tmp.highpasses)
        for j in range(J-1):
            for direc in range(tmp.highpasses[j].shape[2]):              
                g_w.highpasses[j][:, :, direc] =  tmp.highpasses[j][:, :, direc] + g_w.highpasses[j][:, :, direc]
                g_w.highpasses[j][:, :, direc] = softComplexTh(g_w.highpasses[j][:, :, direc], Th)
        
        tmp  = g_w.highpasses[1][:, :, 1]
        Th  = np.median(abs(tmp))/0.6745    
    return DTCWT_Inv(g_w)
            
def denoiseW(w, Th):
    # estimate the theshold 
    
    
    #print('Threshold = {}'.format(Nsig))
    
    J = len(w.highpasses)
#     print('J = {}'.format(J))
    
    for j in range(J-1):
        for direc in range(w.highpasses[j].shape[2]):              
            w.highpasses[j][:, :, direc] = softComplexTh(w.highpasses[j][:, :, direc] , Th)
            
    
    # median filter
    winH = (3,5)
    winV = (5,3)
    winD = (5,5)
    
    for j in range(J-1):
        for direc in range(w.highpasses[j].shape[2]): 
            tmp = w.highpasses[j][:, :, direc]
            if direc == 0 or direc == 5: # Horizontal
                mag = medfilt2d(np.abs(tmp), winH)                
            elif direc == 1 or direc == 4: #Diagonal
                mag = medfilt2d(np.abs(tmp), winD)
            elif direc == 2 or direc == 3: # Vertical
                mag = medfilt2d(np.abs(tmp), winV)
            
            w.highpasses[j][:, :, direc] = np.multiply(mag, np.exp(1j*np.angle(tmp))) 
            
    
    return w

def softComplexTh(w, th):    
    mag = np.abs(w) - th
    mag = np.multiply((mag > 0), mag)
    return np.multiply(mag, np.exp(1j*np.angle(w)))   
    
    
    