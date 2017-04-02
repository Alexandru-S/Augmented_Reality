#pragma once
#ifndef WEBCAM_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define WEBCAM_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iomanip>
#include <string>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int Webcam();


#endif