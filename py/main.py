# import the necessary packages

import numpy as np
import argparse
import datetime
import imutils
import time
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

firstbool = True

cap = cv2.VideoCapture(0)
imgL = cv2.imread('im1.jpg', 0)
imgR = cv2.imread('im2.jpg', 0)
plt.imshow(imgL,'Blues')
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_eye.xml')


fgbg = cv2.createBackgroundSubtractorMOG2()

while(firstbool):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #color = cv2.cvtColor(frame, )
    
    
    if cv2.waitKey(33) == ord('q'):
        print ('pressed Q')
        firstbool = False
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        #face detection code taken from the opencv3 docs
        #http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


#cv2.imshow('gray',gray)
        fgmask = fgbg.apply(frame)
#cv2.imshow('frame',fgmask)
        res = cv2.bitwise_and(frame,frame,mask = fgmask)
        cv2.imshow('frame',res)

#cv2.imshow('frame', frame)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
