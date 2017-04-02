#include "Webcam.h"

//http://answers.opencv.org/question/1/how-can-i-get-frames-from-my-webcam/
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
		if (frame.empty()) break; // end of video stream
		imshow("webcam", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	return 0;
}
