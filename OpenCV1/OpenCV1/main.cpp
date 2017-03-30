/**
* @file Morphology_1.cpp
* @brief Erosion and Dilation sample code
* @author OpenCV team
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <iostream>


using namespace cv;
using namespace std;

int inp;

void printSeparator();
void Video();
int Img();

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

		/*Mat image;
		image = imread("img2.png", CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!image.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
		imshow("Display window", image);                   // Show our image inside it.

		waitKey(0);*/
		Img();

	}

                                        // Wait for a keystroke in the window
	return 0;
}


int Img()
{

	Mat image;
	image = imread("img2.png", CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);


}

void Video() 
{
	printf("Video stitching selected\n");
}


void printSeparator()
{
	printf("-------------------------------------------\n");
}