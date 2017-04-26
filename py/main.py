# import the necessary packages

import numpy as np
import argparse
import datetime
import imutils
import utils
import math
import time
import cv2
import os
import sys
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
#from matchers import matchers

firstbool = True

cap = cv2.VideoCapture(0)

imgL = cv2.imread('im1.jpg', 0)
imgR = cv2.imread('im2.jpg', 0)

Fimg = cv2.imread('img2.jpg', 0)
Simg = cv2.imread('img1.jpg', 0)

videotest = cv2.VideoCapture('video2.mp4')

plt.imshow(imgL,'Blues')
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_eye.xml')

fgbg = cv2.createBackgroundSubtractorMOG2()

choise = input("Press \n1 for face detection with background subtraction\n2 for panorama stitching\n3 for picture stitching \n4 for other picture stitching \n")
print( "you entered", choise)


if choise is '1':
	while(firstbool):
		# Capture frame-by-frame
		if cv2.waitKey(33) == ord('q'):
			print ('pressed Q')
			firstbool = False
			cap.release()
			cv2.destroyAllWindows()
			break
		else:

			ret, frame = cap.read()
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#faces = face_cascade.detectMultiScale(gray, 1.3, 5)


			#face detection code taken from the opencv3 docs using haar cascades
			#http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
			#for (x,y,w,h) in faces:
			#	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			#	roi_gray = gray[y:y+h, x:x+w]
			#	roi_color = frame[y:y+h, x:x+w]
			#	eyes = eye_cascade.detectMultiScale(roi_gray)
			#	for (ex,ey,ew,eh) in eyes:
			#		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

			#fgmask = fgbg.apply(frame)
			#notnot = cv2.bitwise_not(fgmask)

			#res = cv2.bitwise_and(frame,frame,mask = fgmask)
			cv2.imshow('frame', frame)

elif choise is '2':
	print( "option 2" )
	cv2.imshow("frame", imgL)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	


elif choise is '3':
	print( "option 3" )
	count = 0
	seconds = 2
	cap = cv2.VideoCapture('video2.mp4')
	fps = cap.get(cv2.CAP_PROP_FPS) 
	multiplier = fps * seconds
	prev =0
	while(cap.isOpened()):
		
		frameId = int(round(cap.get(1)))
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frameId % multiplier == 0:
			cv2.imshow('frame3', prev)
			cv2.imshow('frame',frame)
			time.sleep(.001)
			prev=frame

		



		count = count + 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	print('complete')


elif choise is '4':
	print("option 4")



stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
