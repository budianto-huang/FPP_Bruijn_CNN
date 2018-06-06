import time
from torch.autograd import Variable
import torch
import numpy as np

def processCNN(netG, Inp):
    time_start = time.clock()
    # prepare for input to CNN
    O = 1

    if len(Inp.shape) == 2:
        C = 1
    else:
        C = Inp.shape[2]

    M, N = Inp.shape[:2]
    inpt = Inp.reshape((O, M, N, C))
    inpt = inpt.transpose((0, 3, 1, 2))

    out = netG(Variable(torch.from_numpy(inpt.astype(np.float32)).cuda()))

    out = out.data.cpu().numpy()
    [O, P, M, N] = out.shape
    out = out.reshape((M, N))
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed * 1000, 'ms')

    return out