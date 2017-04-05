#include "Img.h"

void Window(Mat, String);
void advStitch();


Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;

Mat matMogMask; 
Ptr<BackgroundSubtractor> ptrMog;



struct userdata {
	Mat im;
	vector<Point2f> points;
};

void mouseHandler(int event, int x, int y, int flags, void* data_ptr);


//https://ramsrigoutham.com/2012/11/22/panorama-image-stitching-in-opencv/
int Img()
{
	

	Mat image1;
	Mat image2;
	Mat image12;
	int inp2;

	Mat grayimage1;
	Mat grayimage2;

	ptrMog = createBackgroundSubtractorMOG2();

	printf("please specify which stitching you are using:\n 1: normal stitching \n 2: advanced stitching \n 3: background subtraction \n 4: background augmentation \n");
	cin >> inp2;


	image1 = imread("pic1.jpg", CV_LOAD_IMAGE_COLOR);   
	image2 = imread("pic2.jpg", CV_LOAD_IMAGE_COLOR);   
	image12 = imread("img12.jpg", CV_LOAD_IMAGE_COLOR);

	Mat im_first = imread("first-image.jpg");
	Mat im_scene = imread("times-square.jpg");
	Size size = im_first.size();




	if (inp2 == 1) {

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
	if (inp2 == 2)
	{


		printf("advanced stitching");
		advStitch();
		return 0;
	}
	if (inp2 == 3)
	{

		if (!image1.data || !image2.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		error(image1);

	

		ptrMog->apply(image1, matMogMask);

		rectangle(image12, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);


		imshow("FG Mask MOG 2", matMogMask);
		//imshow("FG Mask MOG qf", image1);
		imshow("image 12", image12);
		waitKey(0);
	}
	else if (inp2 == 4)
	{
		vector<Point2f> pts_src;
		pts_src.push_back(Point2f(0, 0));
		pts_src.push_back(Point2f(size.width - 1, 0));
		pts_src.push_back(Point2f(size.width - 1, size.height - 1));
		pts_src.push_back(Point2f(0, size.height - 1));

		Mat im_temp = im_scene.clone();
		userdata data;
		data.im = im_temp;
		imshow("Image", im_temp);


		cout << "Click on four corners of a billboard and then press ENTER" << endl;
		//set the callback function for any mouse event
		setMouseCallback("Image", mouseHandler, &data);
		waitKey(0);

		// Calculate Homography between source and destination points
		Mat h = findHomography(pts_src, data.points);

		// Warp source image
		warpPerspective(im_first, im_temp, h, im_temp.size());

		// Extract four points from mouse data
		Point pts_dst[4];
		for (int i = 0; i < 4; i++)
		{
			pts_dst[i] = data.points[i];
		}

		// Black out polygonal area in destination image.
		fillConvexPoly(im_scene, pts_dst, 4, Scalar(0), CV_AA);

		im_scene = im_scene + im_temp;

		// Display image.
		imshow("Image", im_scene);
		waitKey(0);
	
	}





}


void mouseHandler(int event, int x, int y, int flags, void* data_ptr)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		userdata *data = ((userdata *)data_ptr);
		circle(data->im, Point(x, y), 3, Scalar(0, 255, 255), 5, CV_AA);
		imshow("Image", data->im);
		if (data->points.size() < 4)
		{
			data->points.push_back(Point2f(x, y));
		}
	}

}


//http://study.marearts.com/2013/11/opencv-stitching-example-stitcher-class.html
void advStitch()
{

	vector< Mat > vImg;
	Mat rImg;

	vImg.push_back(imread("img11.jpg"));
	vImg.push_back(imread("img12.jpg"));
	vImg.push_back(imread("img13.jpg"));
	vImg.push_back(imread("img14.jpg"));


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