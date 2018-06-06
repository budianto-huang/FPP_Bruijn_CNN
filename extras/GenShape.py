import numpy as np
import numpy.random as nrand
#from peaks import peaks

def GenShape(size = 512, fo=3, shapeType = 'cone', noiseLevel = 10.0):
    
    # amplitude of the fringes
    Amp = 255
    
    # Generate mesh grid
    x = np.linspace(-size+1, size, size*2)    
    X, Y  = np.meshgrid(x, x)    
    fs = size / 2
    
    if shapeType == 'cone':
        shape = size - np.sqrt((X**2+Y**2))
        shape = shape * (shape>0)/size * 20
        
        refRad =  2 * np.pi * fo / fs * x - 2*np.pi/3
        print("refRad = ", refRad.shape)
        refRad = np.repeat(refRad, 3, axis=2) 
        print("refRad = ", refRad.shape)        
        
        inpRad = refRad + shape
        
        shapeFringe = np.sin(inpRad)
        refFringe = np.sin(refRad)
    
    elif shapeType == 'wave':
        print('wave')
        shape = peaks(2*size)
        
        shapeFringe = 2*np.pi*fo/fs*X + 3 * shape
#         shapeFringe   = np.tile(shapeFringe, (3,1,1))
#         shapeFringe[0,:,:] = shapeFringe[1,:,:] - 2*np.pi/3
#         shapeFringe[2,:,:] = shapeFringe[1,:,:] + 2*np.pi/3
        
        refFringe   = 2*np.pi*fo/fs*X         
#         refFringe   = np.tile(refFringe, (3,1,1))
#         refFringe[0,:,:] = refFringe[1,:,:] - 2*np.pi/3
#         refFringe[2,:,:] = refFringe[1,:,:] + 2*np.pi/3 
         
        refFringe   = np.sin(refFringe)
        shapeFringe = np.sin(shapeFringe)  
      
    # add bias (original range is [-1, 1]change to [0, 1]
    print("size = ", refFringe.shape)
    shapeFringeZero = (shapeFringe / 2.0) * Amp 
    refFringeZero   = (refFringe / 2.0) * Amp
     
    shapeFringe = shapeFringeZero  + 0.5
    refFringe   = refFringeZero    + 0.5 
     
      
    # add noise
    sigma = float(noiseLevel) 
    print(sigma)
    if sigma > 0.0:
        #noise = np.random.normal(loc=0, scale=sigma, size=3*(2*size)**2).reshape((3, size*2, size*2));
        noise = np.random.normal(loc=0, scale=sigma, size=(2*size)**2).reshape((size*2, size*2));
    else:
        noise = np.zeros((size*2, size*2))
        
    
    
    shapeFringeNoisy = np.clip(np.add(shapeFringe, noise ), 0, 255)
    refFringeNoisy   = np.clip(np.add(refFringe,   noise ), 0, 255)
    
    
#     shapeFringe     = np.transpose(shapeFringe, (1, 2, 0)) 
#     refFringe       = np.transpose(refFringe, (1, 2, 0)) 
#     shapeFringeZero = np.transpose(shapeFringeZero, (1, 2, 0)) 
#     refFringeZero   = np.transpose(refFringeZero, (1, 2, 0)) 
#     shapeFringeNoisy= np.transpose(shapeFringeNoisy, (1, 2, 0)) 
#     refFringeNoisy  = np.transpose(refFringeNoisy, (1, 2, 0))
    
    print("Shape size = ", shapeFringeNoisy.shape, " ", refFringeNoisy.shape)
    print("Generate Shape finished")  
           
    return shape, shapeFringe, refFringe, shapeFringeZero, refFringeZero, shapeFringeNoisy, refFringeNoisy

def peaks(arg1):
    size = arg1 
    X = np.linspace(-3, 3, size)
    x, y  = np.meshgrid(X, X)  
    z =  3.0*(1-x)**2*np.exp(-(x**2) - (y+1)**2) \
        - 10.0*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) \
        - 1/3.0*np.exp(-(x+1)**2 - y**2);
        
    return z 