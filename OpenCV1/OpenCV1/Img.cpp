#include "Img.h"

void Window(Mat, String);


Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;

//https://ramsrigoutham.com/2012/11/22/panorama-image-stitching-in-opencv/
int Img()
{

	Mat image1;
	Mat image2;
	int inp2;

	Mat grayimage1;
	Mat grayimage2;
	printf("please specify which stitching you are using:\n 1: normal stitching \n 2: advanced stitching \n");
	cin >> inp2;
	cout << inp2 << endl;

	if (inp2 == 1) {


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

		warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, image1.rows));
		cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
		image2.copyTo(half);
		//imshow("Result", result);
		Window(result, "result");

		Window(grayimage2, "grey image2");
		Window(img_keypoints_1, "key img 1");
		waitKey(0);
	}
	else if (inp2 = 2)
	{

		printf("advanced stitching");
		return -1;
	}

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