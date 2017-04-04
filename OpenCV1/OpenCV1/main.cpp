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

int printSeparator( int);

int inp1;

int main()
{

	printSeparator(2);
	printf("--- AUGMENTED REALITY REPORT ---\n");
	printSeparator(2);
	printf("please select from the following:\n 1 for video \n 2 for img \n 3 for webcam \n");
	cin >> inp1;
	printSeparator(1);


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




int printSeparator( int lines)
{
	for (int i = 0; i < lines; i++)
	{
		printf("-------------------------------------------\n");
	}
	
	return 0;
}