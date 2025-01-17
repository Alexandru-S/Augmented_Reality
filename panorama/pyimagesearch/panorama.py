# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		(imageB, imageA) = images
		descriptor = cv2.xfeatures2d.SIFT_create()
		(kpsA, featuresA) = descriptor.detectAndCompute(imageA,None)
		kpsA = np.float32([kp.pt for kp in kpsA])
		(kpsB, featuresB) = descriptor.detectAndCompute(imageB, None)
		kpsB = np.float32([kp.pt for kp in kpsB])
		M = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)
		if M is None:
			print('M is none')
			return None
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
			return (result, vis)
		return result

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		rawMatches = flann.knnMatch(featuresA, featuresB, 2)
		matches = []
		for m in rawMatches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		if len(matches) > 4:
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
			return (matches, H, status)
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		return vis
