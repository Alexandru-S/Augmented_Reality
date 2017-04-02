#include "Video.h"

int Video()
{
	string filename = "video1.mp4";
	VideoCapture capture(filename);
	Mat frame;

	if (!capture.isOpened())
		throw "Error when reading video";

	namedWindow("video", 1);
	for (;;)
	{
		capture >> frame;
		if (frame.empty())
		{
			break;
		}
		imshow("video", frame);
		waitKey(20);
		if (waitKey(10) == 27) break;
	}
	return 0;

}