/**
* @OpenCV Project
* @Splicing
* @author Alexandru Sulea
*/


#include "Img.h"
#include "Webcam.h"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int inp;

void printSeparator();
int Video();

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
		imshow("WEBCAM", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	return 0;
}





void printSeparator()
{
	printf("-------------------------------------------\n");
}