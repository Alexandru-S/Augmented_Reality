from pyimagesearch.panorama import Stitcher
#from kivy.app import App
#from kivy.uix.button import Button
import numpy as np
from numpy import pi,sin,cos,mgrid
#from mayavi import mlab
#import vtk
import argparse
import imutils
import cv2

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

ap = argparse.ArgumentParser()

choice =0

print('please choose from the options')
choice = input('1 : basic image stitching\n2 : basic panorama video stitching\n3 : basic panorama stitching from phone camera \n\n')

if choice is '1':

    #http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
    #stitching pipeline code taken from above but modified to work with sift instead
    #of brute force and a few other things
    print('option 1 selected')

    imageA = cv2.imread('ibr.jpg')
    imageB = cv2.imread('ibl.jpg')
    imageA = cv2.resize(imageA, (0, 0), fx=0.5, fy=0.5)
    imageB = cv2.resize(imageB, (0, 0), fx=0.5, fy=0.5)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)

if choice is '2':
    # http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
    # stitching pipeline code taken from above but modified to work with sift instead
    # of brute force and a few other things
    count = 1
    seconds = 5
    prev= 0
    prev_result = 0
    cap = cv2.VideoCapture('video2.mp4', 0)
    capr = cv2.VideoCapture('toright.mp4', 0)
    capl = cv2.VideoCapture('toleft.mp4', 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    multiplier = fps * seconds

    print('option 2 selected')
    cap = cv2.VideoCapture('video2.mp4', 0)


    while (cap.isOpened()):
        key = cv2.waitKey(1) & 0xff
        frameId = int(round(cap.get(1)))
        ret, frame = cap.read()




        if frameId % multiplier == 0:
            if count > 2:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                #cv2.imshow('frame', prev)
                #cv2.imshow('prev', frame)
                stitcher = Stitcher()
                (result, vis) = stitcher.stitch([frame, prev], showMatches=True)
                cv2.imshow('result', result)
                cv2.imshow('vis', vis)

                #(result2, vis2) = stitcher.stitch([prev_result, frame], showMatches=True)

                #cv2.imshow('result2', result2)
                #cv2.imshow('vis2', vis2)
                prev_result= result


            prev=frame

            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('complete')

if choice is '3':

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


    print(str(face_cascade))
    print('choice 3\n')


    img = cv2.imread('ibl.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face detection code taken from the opencv3 docs using haar cascades
    # http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    img1 = cv2.imread('ibr.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # face detection code taken from the opencv3 docs using haar cascades
    # http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    for (x, y, w, h) in faces1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray1 = gray1[y:y + h, x:x + w]
        roi_color1 = img1[y:y + h, x:x + w]
        eyes1 = eye_cascade.detectMultiScale(roi_gray1)
        for (ex, ey, ew, eh) in eyes1:
            cv2.rectangle(roi_color1, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    imageA = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    imageB = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('imgA', img)
    cv2.imshow('imgB', img1)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([img1, img], showMatches=True)

    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)

    cv2.waitKey(0)


  



cv2.waitKey(0)
