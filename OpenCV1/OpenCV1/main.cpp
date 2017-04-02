/**
* @OpenCV Project
* @Splicing
* @author Alexandru Sulea
*/

#include <opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/text.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "Img.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <stdio.h>
#include <iostream>

#include <iostream>
#include <algorithm>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int inp;

void printSeparator();
int Video();
int Webcam();

int main()
{
	int inp1;
	printSeparator();
	printf("--- AUGMENTED REALITY REPORT ---\n");
	printSeparator();
	printf("please select from the following:\n 1 for video \n 2 for img \n 3 for webcam \n");
	cin >> inp1;

	if(inp1==1)
	{ 
		Video();
	}
	else if (inp1 == 2)
	{
		Img();
	}
	else if (inp1 == 3)
	{
		Webcam();
	}
                                     
	return 0;
}

int Video() 
{
	VideoCapture video;
	if (!video.open(0))
	{
		return 0;
	}

	for (;;)
	{
		Mat frame;
		video >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("this is you, smile! :)", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	// the camera will be closed automatically upon exit
	// cap.close();
	return 0;
}


//http://answers.opencv.org/question/1/how-can-i-get-frames-from-my-webcam/
//to open webcaam
int Webcam()
{
	VideoCapture webcam;
	if (!webcam.open(0))
	{
		return 0;
	}

	for (;;)
	{
		Mat frame;
		webcam >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("webcam", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	return 0;
}

void printSeparator()
{
	printf("-------------------------------------------\n");
}