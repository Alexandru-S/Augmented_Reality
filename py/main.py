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
from PIL import Image

#from matchers import matchers

firstbool = True

cap = cv2.VideoCapture(0)

imgL = cv2.imread('p1.jpg', 0)
imgR = cv2.imread('p2.jpg', 0)

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
		#	cv2.imshow('frame', frame)

elif choise is '2':
	print( "option 2" )
	cv2.imshow("frame", imgL)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	


elif choise is '3':

	
	print( "option 3" )
	count = 0
	seconds = 2
	cap = cv2.VideoCapture('video1.mp4',0)
	fps = cap.get(cv2.CAP_PROP_FPS) 
	multiplier = fps * seconds
	prev =0
	prev_des = 0
	prev_kp = 0



	ratio = 0.75
	reprojThresh = 4.0


	sift = cv2.xfeatures2d.SIFT_create()
	#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
	while(cap.isOpened()):
		
		frameId = int(round(cap.get(1)))
		ret, frame = cap.read()
	

		if frameId % multiplier == 0:

			gray_frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY );
			height, width = frame.shape[:2]
			print("width:"+ str(width) + " height:" + str(height) )

			kp, des = sift.detectAndCompute(gray_frame, None)

			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
			search_params = dict(checks=50)  # or pass empty dictionary

			#cv2.drawKeypoints(gray_frame,kp,frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )


			flann = cv2.FlannBasedMatcher(index_params, search_params)
			if (count>2):
				matches = flann.knnMatch(prev_des, des, k=2)

				matchesMask = [[0, 0] for i in range(len(matches))]
				for i, (m, n) in enumerate(matches):
					if m.distance < 0.7 * n.distance:
						matchesMask[i] = [1, 0]

				draw_params = dict(matchColor=(0, 255, 0),
								   singlePointColor=(255, 0, 0),
								   matchesMask=matchesMask,
								   flags=0)

				img3 = cv2.drawMatchesKnn(prev, prev_kp, frame, kp, matches, None, **draw_params)

				dst = np.array([[0, 0],[width - 1, 0],[width - 1, height - 1],[0, height - 1]], dtype="float32")
				rect = np.zeros((4, 2), dtype="float32")
				M = cv2.getPerspectiveTransform(rect, dst)
				warp = cv2.warpPerspective(img3, M, (width*count, height))
				#cv2.warp
				#cv2.imshow('newframe', frame)
				#cv2.imshow('prev', prev)
				img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
				cv2.imshow('img3',img3 )
				warp = cv2.resize(warp, (0, 0), fx=0.25, fy=0.25)
				cv2.imshow('warp', warp)
				#cv2.show()




			#time.sleep(.001)
			prev=frame
			prev_kp = kp
			prev_des = des
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
