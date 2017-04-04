/**
* @OpenCV Project
* @Splicing
* @author Alexandru Sulea
*/

#include "Img.h"
#include "Webcam.h"
#include "Video.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int inp;

void printSeparator();

#ifdef _DEBUG  
#pragma comment(lib, "opencv_core320d.lib")   
#pragma comment(lib, "opencv_imgproc320d.lib")   //MAT processing  
#pragma comment(lib, "opencv_highgui320d.lib")  
#pragma comment(lib, "opencv_stitching320d.lib")

#else  
#pragma comment(lib, "opencv_core320.lib")  
#pragma comment(lib, "opencv_imgproc320.lib")  
#pragma comment(lib, "opencv_highgui320.lib")  
#pragma comment(lib, "opencv_stitching320.lib")
#endif  

int main()
{
	vector< Mat > vImg;
	Mat rImg;

	int inp1;
	printSeparator();
	printf("--- AUGMENTED REALITY REPORT ---\n");
	printSeparator();
	printf("please select from the following:\n 1 for video \n 2 for img \n 3 for webcam \n");
	cin >> inp1;


	if(inp1 == 1)
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




void printSeparator()
{
	printf("-------------------------------------------\n");
}