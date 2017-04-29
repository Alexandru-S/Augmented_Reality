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
    print('option 2 selected')
    cap = cv2.VideoCapture('video2.mp4', 0)
    while (cap.isOpened()):
        frameId = int(round(cap.get(1)))
        ret, frame = cap.read()
        cv2.imshow('wrjgbjbg', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('complete')





cv2.waitKey(0)
