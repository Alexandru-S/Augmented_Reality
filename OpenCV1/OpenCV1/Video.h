#pragma once
#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/text.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>

#include <iomanip>
#include <iostream>

#include <string>
#include <stdio.h>
#include <sstream>
#include <algorithm>
//#include <windows.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int Video();

#endif
