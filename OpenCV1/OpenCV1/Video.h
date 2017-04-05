#pragma once
#ifndef VIDEO_H
#define VIDEO_H
#define CERES_FOUND true

#include <opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


#include "opencv2/text.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/opencv.hpp"
//#include <opencv2/video.hpp>



#include <algorithm>
#include <ctime>
#include <cstdio>
#include <iomanip>
#include <iostream>

//#include <windows.h>
#include <string>
#include <stdio.h>
#include <sstream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int Video();

#endif
