#include "Webcam.h"

//http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
//to open webcam

int inpWebCam;

int Webcam()
{

	printf("please specify which stitching you are using:\n 1: normal webcam \n 2: webcam stitching \n 3:  web cam background subtraction \n");
	cin >> inpWebCam;

	if (inpWebCam == 1)
	{
		VideoCapture webcam;
		if (!webcam.open(0))
		{
			return 0;
		}
		else {
			for (;;)
			{
				Mat frame;
				webcam >> frame;
				if (frame.empty()) break;



				imshow("webcam", frame);
				if (waitKey(10) == 27) break;
			}
		}
	}

	if (inpWebCam == 2)
	{
		VideoCapture cap(0);
		Mat curr, prev;
		//if (!cap.)return -1;
		cap >> curr;
		curr.copyTo(prev);
		while (1)
		{
			cap >> curr;
			imwrite("current.png", curr);
			imwrite("previous.png", prev);
			curr.copyTo(prev);
			if (waitKey(5) == 27)break; //Esc pressed
		}
	}

	return 0;
}
