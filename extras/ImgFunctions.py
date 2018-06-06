import cv2
import numpy as np

def loadAnImage(fname, isColor=0, isFloat=True
                ):
    I = cv2.imread(fname, isColor)
    if isFloat:
        I = I.astype(np.float32)

    return I


def saveAnImage(fname, I):


    if isFloat:
        I = I.astype(np.uint8)

    cv2.imwrite(fname, I)
    return I