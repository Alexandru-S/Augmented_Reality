#pragma once
#ifndef IMG_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define IMG_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int Img();
int error(Mat);

#endif
