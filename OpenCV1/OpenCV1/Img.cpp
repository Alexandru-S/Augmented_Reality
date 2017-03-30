#include "Img.h"

void Window(Mat, String);
int Img()
{

	Mat image1;
	Mat image2;

	Mat grayimage1;
	Mat grayimage2;


	image1 = imread("pic1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	image2 = imread("pic2.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	if (!image1.data || !image2.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	error(image1);

	// Convert to Grayscale
	cvtColor(image1, grayimage1, CV_RGB2GRAY);
	cvtColor(image2, grayimage2, CV_RGB2GRAY);

	namedWindow("Display window", WINDOW_KEEPRATIO);
	resizeWindow("Display window", 600, 600);// Create a window for display.
	imshow("Display window", grayimage1);                   // Show our image inside it.

	namedWindow("Display window 2", WINDOW_NORMAL);
	resizeWindow("Display window 2", 600, 600);// Create a window for display.
	imshow("Display window 2", image2);

	Window(grayimage2 , "grey image2");

	waitKey(0);
}

int error(Mat err) {
	if (!err.data )
	{
		cout << "Error reading mat or image" <<endl;
		return -1;
	}
}


void Window(Mat imageD , String name)
{
	namedWindow(name, WINDOW_KEEPRATIO);
	resizeWindow(name, 600, 600);// Create a window for display.
	imshow(name, imageD);

}