import numpy as np



def dtcwt_denoise(w):
    J = len(w.highpasses)
#     print('J = {}'.format(J))
    
    
    # estimate the theshold 
    tmp  = w.highpasses[1][:, :, 1]
    Nsig = np.median(abs(tmp))/0.6745
    
    #print('Threshold = {}'.format(Nsig))
    
    for j in range(J-1):
        for direc in range(w.highpasses[j].shape[2]):
            wDir = w.highpasses[j][:, :, direc]   
#             if j == 1:         
#                 print((np.abs(wDir) > Nsig))
            absTh =(np.abs(wDir) > Nsig)
            en = np.multiply(np.abs(wDir), absTh)
            ang = np.angle(wDir)
            w.highpasses[j][:, :, direc] = np.multiply(en, np.exp(1j*ang))
            
            
    return w
            
