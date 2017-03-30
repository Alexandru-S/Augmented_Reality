/**
* @file Morphology_1.cpp
* @brief Erosion and Dilation sample code
* @author OpenCV team
*/

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "Img.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <iostream>


#include <iostream>
#include <algorithm>


using namespace cv;
using namespace std;

int inp;

void printSeparator();
void Video();

int main()
{
	int inp1;
	printSeparator();
	printf("please select 1 for video and 2 for img stitching\n");
	cin >> inp1;

	if(inp1==1)
	{ 
	
	}
	else if (inp1 == 2)
	{
		Img();
	}

                                     
	return 0;
}

void Video() 
{
	
}

void printSeparator()
{
	printf("-------------------------------------------\n");
}