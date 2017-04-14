# import the necessary packages

import numpy as np
import argparse
import datetime
import imutils
import time
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from pyimagesearch import panorama as Stitcher


cap = cv2.VideoCapture(0)
imgL = cv2.imread('im1.jpg', 0)
imgR = cv2.imread('im2.jpg', 0)
plt.imshow(imgL,'Blues')

firstbool = True

while(firstbool):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #color = cv2.cvtColor(frame, )
    
    if cv2.waitKey(33) == ord('q'):
        print ('pressed Q')
        firstbool = False
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        cv2.imshow('frame', frame)



stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
