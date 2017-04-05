#include "Video.h"


Mat matVideoMask; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> ptrMog2; //MOG2 Background subtractor
char keyboard; //input from keyboard

Mat frame;
Mat currVid, prevVid;

Mat grayimage1;
Mat grayimage2;



int invid;
int Video()
{
	ptrMog2 = createBackgroundSubtractorMOG2();

	string filename = "video1.mp4";
	string filename2 = "video2.mp4";

	VideoCapture capture(filename);
	VideoCapture capture2(filename2);

	printf("please select from the following:\n 1 for video \n 2 for background subtraction\n 3 for panorama \n");
	cin >> invid;

	if (invid == 1)
	{

		if (!capture.isOpened()) {
			//error in opening the video input
			cerr << "Unable to open video file: " << filename << endl;
			exit(EXIT_FAILURE);
		}
		//read input data. ESC or 'q' for quitting
		while ((char)keyboard != 'q' && (char)keyboard != 27) {

			if (!capture.read(frame)) {
				cerr << "Unable to read next frame." << endl;
				cerr << "Exiting..." << endl;
				exit(EXIT_FAILURE);
			}

			printf("no of frames", frame.size());

			ptrMog2->apply(frame, matVideoMask);

			stringstream ss;
			rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
				cv::Scalar(255, 255, 255), -1);
			ss << capture.get(CAP_PROP_POS_FRAMES);
			string frameNumberString = ss.str();
			putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
				FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

			imshow("Frame", frame);
			//imshow("FG Mask MOG 2", matVideoMask);

			keyboard = waitKey(30);
		}


	}
	if (invid == 2)
	{


		if (!capture.isOpened()) {
			//error in opening the video input
			cerr << "Unable to open video file: " << filename << endl;
			exit(EXIT_FAILURE);
		}
		//read input data. ESC or 'q' for quitting
		while ((char)keyboard != 'q' && (char)keyboard != 27) {

			if (!capture.read(frame)) {
				cerr << "Unable to read next frame." << endl;
				cerr << "Exiting..." << endl;
				exit(EXIT_FAILURE);
			}

			printf("no of frames", frame.size());

			ptrMog2->apply(frame, matVideoMask);

			stringstream ss;
			rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
				cv::Scalar(255, 255, 255), -1);
			ss << capture.get(CAP_PROP_POS_FRAMES);
			string frameNumberString = ss.str();
			putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
				FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

			imshow("Frame", frame);
			imshow("FG Mask MOG 2", matVideoMask);

			keyboard = waitKey(30);
		}
	
	
	
	
	
	}

	
	if (invid == 3)
	{
		int frameCnt = capture2.get(CV_CAP_PROP_FRAME_COUNT);
		cout << frameCnt << endl;

		if (!capture2.isOpened()) {
			cerr << "Unable to open video file: " << filename << endl;
			exit(EXIT_FAILURE);
		}

		Mat prev;
		Mat result;
		Mat curr;

		Mat greyimg;
		Mat greyprev;

		Mat image;

		capture2 >> curr;
		curr.copyTo(prev);

		vector< Mat > vImg;
		Mat rImg;

		capture2.set(CV_CAP_PROP_FRAME_WIDTH, 600);
		capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 320);

		while ((char)keyboard != 'q' && (char)keyboard != 27) {

			if (!capture2.read(frame)) {
				cerr << "Unable to read next frame." << endl;
				cerr << "Exiting..." << endl;
				exit(EXIT_FAILURE);
			}

			capture2.read(image);

		


			stringstream ss;
			rectangle(image, cv::Point(10, 2), cv::Point(100, 20),
				cv::Scalar(255, 255, 255), -1);
			ss << capture2.get(CAP_PROP_POS_FRAMES);
			string frameNumberString = ss.str();
			putText(image, frameNumberString.c_str(), cv::Point(15, 15),
				FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));


			resize(image, image, Size(600, 320), 0, 0, INTER_CUBIC);
	imshow("image", image);
			resize(prev, prev, Size(600, 320), 0, 0, INTER_CUBIC);
			imshow("prev", prev);

			// Convert to Grayscale
			//Mat testPrev = prev;
			//Mat testimage = image;

			//cvtColor(prev, grayimage2, CV_RGB2GRAY);
			//cvtColor(image, grayimage1, CV_RGB2GRAY);

			Stitcher stitcher = Stitcher::createDefault();

			unsigned long AAtime = 0, BBtime = 0;
			AAtime = getTickCount();

			Stitcher::Status status = stitcher.stitch(vImg, rImg);

			BBtime = getTickCount(); //check processing time 
			printf("%.2lf sec \n", (BBtime - AAtime) / getTickFrequency()); //check processing time

			if (Stitcher::OK == status)
				imshow("Stitching Result", rImg);
			else
				printf("Stitching fail.");


			image.copyTo(prev);
			keyboard = waitKey(1);

		}
	}

	if (invid == 4)
	{
		int frameCnt = capture2.get(CV_CAP_PROP_FRAME_COUNT);
		cout << frameCnt << endl;

		if (!capture2.isOpened()) {
			cerr << "Unable to open video file: " << filename << endl;
			exit(EXIT_FAILURE);
		}

		Mat prev;
		Mat result;
		Mat curr;


		Mat greyimg;
		Mat greyprev;

		
		Mat image;

		capture2 >> curr;
		curr.copyTo(prev);

		//imshow("1", curr);
		//imshow("2", prev);

		capture2.set(CV_CAP_PROP_FRAME_WIDTH, 600);
		capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 320);

		while ((char)keyboard != 'q' && (char)keyboard != 27) {

			if (!capture2.read(frame)) {
				cerr << "Unable to read next frame." << endl;
				cerr << "Exiting..." << endl;
				exit(EXIT_FAILURE);
			}
			
			capture2.read(image);
		
			
			stringstream ss;
			rectangle(image, cv::Point(10, 2), cv::Point(100, 20),
				cv::Scalar(255, 255, 255), -1);
			ss << capture2.get(CAP_PROP_POS_FRAMES);
			string frameNumberString = ss.str();
			putText(image, frameNumberString.c_str(), cv::Point(15, 15),
				FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));


			resize(image, image, Size(600, 320), 0, 0, INTER_CUBIC);
			imshow("image", image);
			resize(prev, prev, Size(600, 320), 0, 0, INTER_CUBIC);
			imshow("prev", prev);

			// Convert to Grayscale
			Mat testPrev = prev;
			Mat testimage = image;

			cvtColor(prev, grayimage2, CV_RGB2GRAY);
			cvtColor(image, grayimage1, CV_RGB2GRAY);

			//http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_detection/feature_detection.html
			//-- Step 1: Detect the keypoints using SURF Detector
			int minHessian = 400;

			Ptr<SURF> detector = SURF::create(minHessian);

			std::vector<KeyPoint> keypoints_1, keypoints_2;

			detector->detect(grayimage1, keypoints_1);
			detector->detect(grayimage2, keypoints_2);

			//-- Draw keypoints
			Mat img_keypoints_1; Mat img_keypoints_2;

			drawKeypoints(grayimage1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			drawKeypoints(grayimage2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);


			// Extractor
			Ptr<SURF> extractor = SURF::create();

			Mat descriptors_1, descriptors_2;

			extractor->compute(grayimage1, keypoints_1, descriptors_1);
			extractor->compute(grayimage2, keypoints_2, descriptors_2);

			//-- Step 3: Matching descriptor vectors using FLANN matcher
			//FlannBasedMatcher matcher;
			FlannBasedMatcher matcher;
			std::vector< DMatch > matches;
			matcher.match(descriptors_1, descriptors_2, matches);

			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < descriptors_1.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			printf("-- Max dist : %f \n", max_dist);
			printf("-- Min dist : %f \n", min_dist);

			// --Use only "good" matches(i.e.whose distance is less than 3 * min_dist)
			std::vector< DMatch > good_matches;

			for (int i = 0; i < descriptors_1.rows; i++)
			{
				if (matches[i].distance < 3 * min_dist)
				{
					good_matches.push_back(matches[i]);
				}
			}
			std::vector< Point2f > obj;
			std::vector< Point2f > scene;

			for (int i = 0; i < good_matches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
				scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
			}

			Mat H = findHomography(obj, scene, CV_RANSAC);
			// Use the Homography Matrix to warp the images
			cv::Mat result;

			warpPerspective(testPrev, result, H, cv::Size(testPrev.cols + testimage.cols, testPrev.rows));
			cv::Mat half(result, cv::Rect(0, 0, testimage.cols, testimage.rows));
			testimage.copyTo(half);
			//imshow("Result", result);
			imshow("result",result);

		    imshow("gray",grayimage2);
			imshow("img_key",img_keypoints_1);
			

			image.copyTo(prev);
			keyboard = waitKey(30);
		}
	}

	capture.release();
}