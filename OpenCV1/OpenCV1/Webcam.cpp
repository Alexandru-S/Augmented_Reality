#include "Webcam.h"

//http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
//to open webcam
int Webcam()
{
	VideoCapture webcam;
	if (!webcam.open(0))
	{
		return 0;
	}

	for (;;)
	{
		Mat frame;
		webcam >> frame;
		if (frame.empty()) break; 
		imshow("webcam", frame);
		if (waitKey(10) == 27) break; 
	}
	return 0;
}
