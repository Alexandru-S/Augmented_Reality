#pragma once
#ifndef IMG_H
#define IMG_H
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
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"

//#include <opencv2/sfm.hpp>
//#include <opencv2/viz.hpp>


#include <algorithm>
#include <ctime>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sstream>





using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
//using namespace cv::sfm;

int Img();
int error(Mat);

#endif
