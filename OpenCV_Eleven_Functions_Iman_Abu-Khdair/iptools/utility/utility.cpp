#include "utility.h"
#include "../iptools/core.h"
#include "../image/image.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sstream>
#include <math.h>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
// #include <opencv2/datasets/gr_skig.hpp>

#define MAXRGB 255
#define MINRGB 0

using namespace cv;

std::string utility::intToString(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int utility::checkValue(int value)
{
	if (value > MAXRGB)
		return MAXRGB;
	if (value < MINRGB)
		return MINRGB;
	return value;
}

/*-----------------------------------------------------------------------**/
void utility::addGrey(image &src, image &tgt, int value)
{
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			tgt.setPixel(i,j,checkValue(src.getPixel(i,j)+value)); 
		}
}

/*-----------------------------------------------------------------------**/
void utility::binarize(image &src, image &tgt, int threshold)
{
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			if (src.getPixel(i,j) < threshold)
				tgt.setPixel(i,j,MINRGB);
			else
				tgt.setPixel(i,j,MAXRGB);
		}
	}
}

/*-----------------------------------------------------------------------**/
void utility::scale(image &src, image &tgt, float ratio)
{
	int rows = (int)((float)src.getNumberOfRows() * ratio);
	int cols  = (int)((float)src.getNumberOfColumns() * ratio);
	tgt.resize(rows, cols);
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{	
			int i2 = (int)floor((float)i/ratio);
			int j2 = (int)floor((float)j/ratio);
			if (ratio == 2) {
				tgt.setPixel(i,j,checkValue(src.getPixel(i2,j2)));
			}

			if (ratio == 0.5) {
				int value = src.getPixel(i2,j2) + src.getPixel(i2,j2+1) + src.getPixel(i2+1,j2) + src.getPixel(i2+1,j2+1);
				tgt.setPixel(i,j,checkValue(value/4));
			}
		}
	}
}

/*-----------------------------------------------------------------------**/
void utility::cv_gray(Mat &src, Mat &tgt)
{
	cvtColor(src, tgt, COLOR_BGR2GRAY);
}

/*-----------------------------------------------------------------------**/
void utility::cv_avgblur(Mat &src, Mat &tgt, int WindowSize)
{
	blur(src,tgt,Size(WindowSize,WindowSize));
}



/*---------------------------------------------------------------------------------------------------------------**
***								PART 1: ADVANCED EDGE DETECTION WITH OPENCV										***
*---------------------------------------------------------------------------------------------------------------**/
/*----------------------- GAUSSIAN SMOOTHING WITH 5X5 KERNEL SIZE -----------------------**/
void utility::gausobel5(Mat &src, Mat &tgt) 
{
	Mat temp;
	// GaussianBlur(src, temp, Size(3, 3), 0, 0);	// output file: jackie_gaus3.ppm	parameter.txt: baboon.pgm baboon_gaus3.ppm opencv gausobel5
	GaussianBlur(src, temp, Size(5, 5), 0, 0);	// output file: jackie_gaus5.ppm	parameter.txt: baboon.pgm baboon_gaus5.ppm opencv gausobel5
	// GaussianBlur(src, temp, Size(9, 9), 0, 0);	// output file: jackie_gaus9.ppm	parameter.txt: baboon.pgm baboon_gaus9.ppm opencv gausobel5
	Sobel(temp, tgt, CV_32F, 1, 1, 5);
}


/*---------------------------------------------------------------------------------------------------------------**
***								PART 2: ADVANCED EDGE DETECTION WITH OPENCV										***
*---------------------------------------------------------------------------------------------------------------**/

/*---------------------------------------------------------------------------------------------------------------**
***									PART A: SOBEL OPERATOR USING OPENCV											***
*---------------------------------------------------------------------------------------------------------------**/

void utility::cv_extraCredit(Mat &src, Mat &tgt) 
{
	Mat temp, converted; 
	GaussianBlur(src, temp, Size(3, 3), 5, 0);
	cvtColor(temp, converted, COLOR_BGR2HSV);


	Mat channel[3];
	split(converted, channel);

	Canny(channel[0], channel[0], 100, 200);
	Canny(channel[1], channel[1], 100, 200);
	Canny(channel[2], channel[2], 100, 200);

	merge(channel, 3, tgt);

}// end cv_extraCredit() function


void utility::cv_sobel3(Mat &src, Mat &tgt)
{
	Mat srcGrey, temp;
	cvtColor(src, srcGrey, COLOR_BGR2GRAY);
	Sobel(srcGrey, temp, CV_32F, 1, 1, 3);

	double min, max;
    minMaxLoc(temp, &min, &max); //find minimum and maximum intensities
    temp.convertTo(tgt, CV_8U, 255.0/(max - min), -min * 255.0/(max - min));

}// end cv_sobel3() function


// void utility::cv_sobel5(Mat &src, Mat &tgt, int XL, int YL, int xLen, int yLen)
void utility::cv_sobel5(Mat &src, Mat &tgt)
{
	Mat srcGrey, temp;
	cvtColor(src, srcGrey, COLOR_BGR2GRAY);
	Sobel(srcGrey, temp, CV_32F, 1, 1, 5);

	double min, max;
    minMaxLoc(temp, &min, &max); //find minimum and maximum intensities
    temp.convertTo(tgt, CV_8U, 255.0/(max - min), -min * 255.0/(max - min));

}// end cv_sobel5() function


/*----------------------- CANNY TECHNIQUE -----------------------**/
void utility::cv_canny(Mat &src, Mat &tgt) 
{
	// variables for dimensions
	// int width = xLen + XL;
	// int height = yLen + YL;

	Mat temp;
	GaussianBlur(src, temp, Size(5, 5), 7, 0);	//shrek-canny-5		
	Canny(temp, tgt, 50, 75);

}// end cv_canny() function



/*---------------------------------------------------------------------------------------------------------------**
***													HOMEWORK 4													***
***									PART 1: HISTOGRAM MODIFICATION WITH OPENCV									***
*---------------------------------------------------------------------------------------------------------------**/
void utility::histostretch(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	normalize(grayedImage, tgt, 0, 255, NORM_MINMAX);

}// end histostretch() function


void utility::histoequ(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	equalizeHist(grayedImage, tgt);

}// end histoequ() function





/*---------------------------------------------------------------------------------------------------------------**
***										PART 2: HOUGH TRANSFORM WITH OPENCV										***
*---------------------------------------------------------------------------------------------------------------**/
// Hough Transform for circles 
void utility::houghcir(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage, blurred;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	// medianBlur(grayedImage, blurred, 5);
	// GaussianBlur(grayedImage, blurred, Size(9, 9), 2, 2);	
	// Canny(blurred, temp, 50, 75);



	vector<Vec3f> circles;

    HoughCircles(grayedImage, circles, HOUGH_GRADIENT,
		  2, 1, grayedImage.rows);
          //tgt.rows()/2,   // accumulator resolution (size of the image / 2)
          //3,  // minimum distance between two circles
          //100, // Canny high threshold
          //20, // minimum number of votes
          //1, 50); // min and max radius


	std::vector<Vec3f>::const_iterator itc = circles.begin();

	while (itc!=circles.end()) {

         cv::circle(grayedImage,
            cv::Point((*itc)[0], (*itc)[1]), // circle centre
            (*itc)[2],       // circle radius
            cv::Scalar(255), // color
            2);              // thickness

         ++itc;
       }

	tgt = grayedImage.clone();

}// end houghcir() function4


// Canny OpenCV module to produce the input for the Hough Transform
void utility::cannyhough(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage, blurred, temp;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	// GaussianBlur(grayedImage, blurred, Size(3, 3), 0, 0);
	medianBlur(grayedImage, blurred, 3);
	Canny(grayedImage, temp, 50, 75);

	vector<Vec3f> circles;
	HoughCircles(temp, circles, HOUGH_GRADIENT, 2, 1, temp.rows);
	std::vector<Vec3f>::const_iterator itc = circles.begin();

	while (itc != circles.end()) {
		cv::circle(temp,
		cv::Point((*itc)[0], (*itc)[1]),
		(*itc)[2],       
		cv::Scalar(255),
		2);              

				++itc;
	}
	tgt = temp.clone();


}// end cannyhough() function


// Smooth the image using Gaussian smoothing prior to edge detection
void utility::smoothedge(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage, blurred, temp;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	GaussianBlur(grayedImage, blurred, Size(5, 5), 0, 0);
	Canny(grayedImage, temp, 50, 75);

	vector<Vec3f> circles;
	HoughCircles(temp, circles, HOUGH_GRADIENT, 2, 1, temp.rows);
	std::vector<Vec3f>::const_iterator itc = circles.begin();

	while (itc != circles.end()) {
		cv::circle(temp,
		cv::Point((*itc)[0], (*itc)[1]),
		(*itc)[2],       
		cv::Scalar(255),
		2);              

				++itc;
	}
	tgt = temp.clone();

}// end smoothedge() function





/*---------------------------------------------------------------------------------------------------------------**
***									PART 3: COMBINING OPERATIONS WITH OPENCV									***
*---------------------------------------------------------------------------------------------------------------**/
// Otsu method (same as from Homework 3)
void utility::cv_otsu(Mat &src, Mat &tgt) 
{
	Mat makeGray, temp;
	cvtColor(src, makeGray, COLOR_BGR2GRAY);
	Sobel(makeGray, temp, -1, 1, 1, 3);
	threshold(temp, tgt, 0, 255, THRESH_OTSU);

}// end cv_otsu() function


// Otsu method and histogram equilization combined
void utility::otsuhisto(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage, temp;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	equalizeHist(grayedImage, temp);
	Sobel(temp, temp, -1, 1, 1, 3);
	threshold(temp, tgt, 0, 255, THRESH_OTSU);

}// end otsuhisto() function


// Gaussian smoothing followed by Otsu method 
void utility::otsugau(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage, temp;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	GaussianBlur(grayedImage, temp, Size(5, 5), 0, 0);	// output file: jackie_gaus5.ppm	parameter.txt: baboon.pgm baboon_gaus5.ppm opencv gausobel5
	// Sobel(temp, temp, CV_32F, 1, 1, 5);
	Sobel(temp, temp, -1, 1, 1, 3);
	threshold(temp, tgt, 0, 255, THRESH_OTSU);

}// end otsugau() function


// Combination of Gaussian smoothing and then histogram equilization and then the Otsu method
void utility::combine(cv::Mat &src, cv::Mat &tgt)
{
	Mat grayedImage, temp;
	cvtColor(src, grayedImage, COLOR_BGR2GRAY);
	GaussianBlur(grayedImage, temp, Size(5, 5), 0, 0);
	equalizeHist(temp, temp);
	threshold(temp, tgt, 0, 255, THRESH_OTSU);


}// end combine() function


// void utility::subtract(cv::Mat &src1, cv::Mat &src2, cv::Mat &tgt)
// {
	// Mat dst, mask;
	// subtract(src1, src2, tgt);
	// tgt = src1 - src2;


// }// end subtract() function

/*---------------------------------------------------------------------------------------------------------------**
***									PART 4: ADVANCED OPERATIONS (EXTRA CREDIT)									***
*---------------------------------------------------------------------------------------------------------------**/
void utility::QRcode(cv::Mat &src, cv::Mat &tgt)
{
	// Mat middle, end;
	// QRCodeDetector qrCode = QRCodeDetector();
	// Mat points, rectImage;
	// int n = points.rows;
  	// string data = qrCode.detectAndDecode(src, points, rectImage);
  	// if(data.length()>0){
  
	// 	cout << "Data after decoding: " << data << endl;
	// 	// display(getImage, points);
	// 	for(int i = 0 ; i < n ; i++)
	// 	{
	// 		line(src, Point2i(points.at<float>(i,0), points.at<float>(i,1)),
	// 		Point2i(points.at<float>((i+1)%n,0), points.at<float>((i+1)%n,1)), Scalar(255,0,0), 3);
	// 	}
	
    // 	rectImage.convertTo(rectImage, CV_8UC3);
	// }

	// qrCode.detectAndDecode(src, middle, end);
	src.convertTo(tgt, -1, 1, 1);

}// end QRcode() function


void utility::QRcodepre(cv::Mat &src, cv::Mat &tgt)
{
	// Mat middle, end;
	// QRCodeDetector qrCode = QRCodeDetector();
	// Mat points, rectImage;
	// int n = points.rows;
  	// string data = qrCode.detectAndDecode(src, points, rectImage);
  	// if(data.length()>0){
  
	// 	cout << "Data after decoding: " << data << endl;
	// 	// display(getImage, points);
	// 	for(int i = 0 ; i < n ; i++)
	// 	{
	// 		line(src, Point2i(points.at<float>(i,0), points.at<float>(i,1)),
	// 		Point2i(points.at<float>((i+1)%n,0), points.at<float>((i+1)%n,1)), Scalar(255,0,0), 3);
	// 	}
	
    // 	rectImage.convertTo(rectImage, CV_8UC3);
	// }

	// qrCode.detectAndDecode(src, middle, end);
	cvtColor(src, tgt, COLOR_BGR2GRAY);


}// end QRcodepre() function