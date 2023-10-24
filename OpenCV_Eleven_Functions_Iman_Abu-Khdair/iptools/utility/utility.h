#ifndef UTILITY_H
#define UTILITY_H

#include "../image/image.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <math.h>

class utility
{
	public:
		utility();
		virtual ~utility();
		static std::string intToString(int number);
		static int checkValue(int value);
		static void addGrey(image &src, image &tgt, int value);
		static void binarize(image &src, image &tgt, int threshold);
		static void scale(image &src, image &tgt, float ratio);
		static void cv_gray(cv::Mat &src, cv::Mat &tgt);
		static void cv_avgblur(cv::Mat &src, cv::Mat &tgt, int WindowSize);

/************************ ADDED FUNCTIONS FOR OPENCV (SOBEL, OTSU, AND CANNY) ************************/
		static void gausobel5(cv::Mat &src, cv::Mat &tgt);

/************************ ADDED FUNCTIONS FOR OPENCV (SOBEL, OTSU, AND CANNY) ************************/
		static void cv_sobel3(cv::Mat &src, cv::Mat &tgt);
		static void cv_sobel5(cv::Mat &src, cv::Mat &tgt);
		
		static void cv_canny(cv::Mat &src, cv::Mat &tgt);
		static void cv_extraCredit(cv::Mat &src, cv::Mat &tgt);
		// static void cv_sobel3(cv::Mat &src, cv::Mat &tgt, int X1, int Y1, int xLen, int yLen);
		// static void cv_sobel5(cv::Mat &src, cv::Mat &tgt, int X1, int Y1, int xLen, int yLen);
		// static void cv_otsu(cv::Mat &src, cv::Mat &tgt, int X1, int Y1, int xLen, int yLen);
		// static void cv_canny(cv::Mat &src, cv::Mat &tgt, int X1, int Y1, int xLen, int yLen);


/*---------------------------------------------------------------------------------------------------------------**
***									PART 1: HISTOGRAM MODIFICATION WITH OPENCV									***
*---------------------------------------------------------------------------------------------------------------**/
		static void histostretch(cv::Mat &src, cv::Mat &tgt);
		static void histoequ(cv::Mat &src, cv::Mat &tgt);


/*---------------------------------------------------------------------------------------------------------------**
***										PART 2: HOUGH TRANSFORM WITH OPENCV										***
*---------------------------------------------------------------------------------------------------------------**/
		static void houghcir(cv::Mat &src, cv::Mat &tgt);
		static void cannyhough(cv::Mat &src, cv::Mat &tgt);
		static void smoothedge(cv::Mat &src, cv::Mat &tgt);

/*---------------------------------------------------------------------------------------------------------------**
***									PART 3: COMBINING OPERATIONS WITH OPENCV									***
*---------------------------------------------------------------------------------------------------------------**/
		static void cv_otsu(cv::Mat &src, cv::Mat &tgt);
		static void otsuhisto(cv::Mat &src, cv::Mat &tgt);
		static void otsugau(cv::Mat &src, cv::Mat &tgt);
		static void combine(cv::Mat &src, cv::Mat &tgt);
		// static void subtract(cv::Mat &src1, cv::Mat &src2, cv::Mat &tgt);


/*---------------------------------------------------------------------------------------------------------------**
***									PART 4: ADVANCED OPERATIONS (EXTRA CREDIT)									***
*---------------------------------------------------------------------------------------------------------------**/
		static void QRcode(cv::Mat &src, cv::Mat &tgt);
		static void QRcodepre(cv::Mat &src, cv::Mat &tgt);

};

#endif

