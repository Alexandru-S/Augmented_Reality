from pyimagesearch.panorama import Stitcher
import numpy as np
import argparse
import imutils
import cv2

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

ap = argparse.ArgumentParser()

choice =0

print('please choose from the options')
choice = input('1 : basic image stitching\n2 : basic panorama stitching \n\n')

if choice is '1':
    print('option 1 selected')

    imageA = cv2.imread('images/bryce_left_01.png')
    imageB = cv2.imread('images/bryce_right_01.png')
    imageA = cv2.resize(imageA, (0, 0), fx=0.5, fy=0.5)
    imageB = cv2.resize(imageB, (0, 0), fx=0.5, fy=0.5)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)

if choice is '2':
    count = 1
    seconds = 2
    prev= 0
    cap = cv2.VideoCapture('video2.mp4', 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    multiplier = fps * seconds

    print('option 2 selected')
    cap = cv2.VideoCapture('video2.mp4', 0)

    imageA = cv2.imread('images/bryce_left_01.png')
    imageB = cv2.imread('images/bryce_right_01.png')
    imageA = cv2.resize(imageA, (0, 0), fx=0.5, fy=0.5)
    imageB = cv2.resize(imageB, (0, 0), fx=0.5, fy=0.5)


    while (cap.isOpened()):
        frameId = int(round(cap.get(1)))
        ret, frame = cap.read()

        if frameId % multiplier == 0:
            if count > 2:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('frame', prev)
                cv2.imshow('prev', frame)
                stitcher = Stitcher()
                (result, vis) = stitcher.stitch([frame, prev], showMatches=True)

                cv2.imshow('result', result)
                cv2.imshow('vis', vis)

            prev=frame
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('complete')
cv2.waitKey(0)
