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

# from matchers import matchers

firstbool = True

cap = cv2.VideoCapture(0)

imgL = cv2.imread('p1.jpg', 0)
imgR = cv2.imread('p2.jpg', 0)

Fimg = cv2.imread('img2.jpg', 0)
Simg = cv2.imread('img1.jpg', 0)

videotest = cv2.VideoCapture('video2.mp4')

plt.imshow(imgL, 'Blues')
face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

fgbg = cv2.createBackgroundSubtractorMOG2()

choise = input(
    "Press \n1 for face detection with background subtraction\n2 for panorama stitching\n3 for picture stitching \n4 for other picture stitching \n")
print("you entered", choise)

if choise is '1':
    while (firstbool):
        # Capture frame-by-frame
        if cv2.waitKey(33) == ord('q'):
            print('pressed Q')
            firstbool = False
            cap.release()
            cv2.destroyAllWindows()
            break
        else:

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)


            # face detection code taken from the opencv3 docs using haar cascades
            # http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
            # for (x,y,w,h) in faces:
            #	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #	roi_gray = gray[y:y+h, x:x+w]
            #	roi_color = frame[y:y+h, x:x+w]
            #	eyes = eye_cascade.detectMultiScale(roi_gray)
            #	for (ex,ey,ew,eh) in eyes:
            #		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            # fgmask = fgbg.apply(frame)
            # notnot = cv2.bitwise_not(fgmask)

            # res = cv2.bitwise_and(frame,frame,mask = fgmask)
            #	cv2.imshow('frame', frame)

elif choise is '2':
    print("option 2")


    def drawMatches( imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]

        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        print('222222222222222222222222222222222222222222222')

        vis[0:hA, 0:wA] = imageA

        vis[0:hB, wA:] = imageB
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis



    def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
            return (matches, H, status)
        return None


    def detectAndDescribe( image):
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)


    def stitch( images1,images2, ratio=0.75, reprojThresh=4.0,showMatches=False):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        imageA = images1
        imageB = images2
        (kpsA, featuresA) = detectAndDescribe(imageA)
        (kpsB, featuresB) = detectAndDescribe(imageB)
        M = matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            return None
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        if showMatches:
            vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)
        return result


    imageA = cv2.imread('p1.jpg', 0)
    imageB = cv2.imread('p2.jpg', 0)

    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)

    cv2.imshow('A', imageA)
    cv2.imshow('B', imageB)

    (result, vis) = stitch(imageA, imageB, ratio=0.75, reprojThresh=4.0,showMatches=True)




    cv2.waitKey(0)
    cv2.destroyAllWindows()




elif choise is '3':
    print("option 3")
    count = 1
    count2 = 0
    seconds = 2
    cap = cv2.VideoCapture('video2.mp4', 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    multiplier = fps * seconds
    prev = 0
    prev_des = 0
    prev_kp = 0
    ratio = 0.75
    reprojThresh = 4.0
    featuresB = 0
    featuresA = 0
    ptsA =0
    H=0
    status=0

    sift = cv2.xfeatures2d.SIFT_create()
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    # http: // www.pyimagesearch.com / 2014 / 05 / 05 / building - pokedex - python - opencv - perspective - warping - step - 5 - 6 /
    # http://www.pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv/
    # http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

    result = 0
    result_gry = None
    sift = cv2.xfeatures2d.SIFT_create()

    while (cap.isOpened()):

        frameId = int(round(cap.get(1)))
        ret, frame = cap.read()

        if frameId % multiplier == 0:

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('qfeeq',gray_frame)

            if not ret:
                break

            if (result is 0):
                print('fefefef')
                result = frame

            else:
                features0 = sift.detectAndCompute(result_gry, None)
                #features1 = sift.detectAndCompute(gray_frame, None)
                #knn = 5
                #lowe = 0.7

                #keypoints0, descriptors0 = features0
                #keypoints1, descriptors1 = features1

                #matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)
                #logger.debug('finding correspondence')






                # height, width = frame.shape[:2]

            #kp, des = sift.detectAndCompute(gray_frame, None)

            #FLANN_INDEX_KDTREE = 0
            #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            #search_params = dict(checks=50)  # or pass empty dictionary
            #flann = cv2.FlannBasedMatcher(index_params, search_params)

            #if (count > 2):
            #    matches = flann.knnMatch(prev_des, des, k=2)
            #    matchesMask = [[0, 0] for i in range(len(matches))]
            #    for i, (m, n) in enumerate(matches):
            #        if m.distance < 0.7 * n.distance:
            #            matchesMask[i] = [1, 0]

             #   draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask,flags=0)

                #img3 = cv2.drawMatchesKnn(prev, prev_kp, frame, kp, matches, None, **draw_params)
                #img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
                #cv2.imshow('img3', img3)

               # descriptor = cv2.xfeatures2d.SIFT_create()
               # (kps, featuresA) = descriptor.detectAndCompute(frame, None)
               # kps = np.float32([kp.pt for kp in kps])
               # matcher = cv2.DescriptorMatcher_create("BruteForce")
               # rawMatches = matcher.knnMatch(des, prev_des, 2)
               # matches2 = []
               # prev_kp = np.float32([k.pt for k in prev_kp])

                #for m in rawMatches:
                  #  if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                 #       matches2.append((m[0].trainIdx, m[0].queryIdx))
                #if len(matches2) > 4:
                   # ptsA = np.float32([prev_kp[i-i] for (_, i) in matches2])
                   # print('ptsA' + str(ptsA))

                    #if(count2>1):
                        #(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
                        #M = (matches2,H, status)
                        #if M is None:
                        #    print("nada")
                       # cv2.cachedH = M[1]
                      #  print("++++++++++")
                     #   result = cv2.warpPerspective(frame, cv2.cachedH,(frame.shape[1] + prev.shape[1], frame.shape[0]))
                    #    cv2.imshow("result", result)
                   # count2+=1



          #  prev_kp = kp
            prev = frame
           # prev_des = des
            ptsB =ptsA
            featuresB = featuresA
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
