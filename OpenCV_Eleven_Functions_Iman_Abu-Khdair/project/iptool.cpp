#include "../iptools/core.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sstream>
#include <math.h>


using namespace std;

#define MAXLEN 256

int main (int argc, char** argv)
{
	image src, tgt;
	FILE *fp;
	char str[MAXLEN];
	char infile[MAXLEN];
	char outfile[MAXLEN];
	char *pch;
	
	if ((fp = fopen(argv[1],"r")) == NULL) {
		fprintf(stderr, "Can't open file: %s\n", argv[1]);
		exit(1);
	}

	while(fgets(str,MAXLEN,fp) != NULL) {
		if (strncmp(str,"#",1)==0) continue;
		int NumParameters = 0;
		
		pch = strtok(str, " ");
		strcpy(infile, pch);

		pch = strtok(NULL, " ");
		strcpy(outfile, pch);

		pch = strtok(NULL, " ");
		if (strncmp(pch,"opencv",6)==0) {
			cv::Mat I = cv::imread(infile);
			cv::Mat I2;
			cv::Mat I_2 = cv::imread(infile);
			
			if( I.empty()) {
				cout << "Could not open or find the image: " << infile << endl;
				return -1;
			}
			
			pch = strtok(NULL, " ");
			if (strncmp(pch,"gray",4)==0) {
				utility::cv_gray(I, I2);
			}
/************************ HW 3: ADDED FUNCTIONS ************************/
			else if (strncmp(pch,"gausobel5",8)==0) {
				utility::gausobel5(I, I2);
			}
			else if(strncmp(pch,"sobel3",8)==0){
				utility::cv_sobel3(I, I2);
			}
			else if(strncmp(pch,"sobel5",8)==0){
				utility::cv_sobel5(I, I2);
			}
			else if(strncmp(pch,"canny",8)==0){
				utility::cv_canny(I, I2);
			}
			// else if(strncmp(pch,"otsu",8)==0){
			// 	utility::cv_otsu(I, I2);
			// }
			else if(strncmp(pch,"extraCredit",8)==0){
				utility::cv_extraCredit(I, I2);
			}

			else if (strncmp(pch,"blur_avg",8)==0) {
				pch = strtok(NULL, " ");
				utility::cv_avgblur(I, I2, atoi(pch));
			}
/****************************************** HW 4: ADDED FUNCTIONS ******************************************/
/*---------------------------------------------------------------------------------------------------------------**
***									PART 1: HISTOGRAM MODIFICATION WITH OPENCV									***
*---------------------------------------------------------------------------------------------------------------**/
			else if(strncmp(pch,"histostretch",8)==0){
				utility::histostretch(I, I2);
			}

			else if(strncmp(pch,"histoequ",8)==0){
				utility::histoequ(I, I2);
			}

/*---------------------------------------------------------------------------------------------------------------**
***										PART 2: HOUGH TRANSFORM WITH OPENCV										***
*---------------------------------------------------------------------------------------------------------------**/
			else if(strncmp(pch,"houghcir",8)==0){
				utility::houghcir(I, I2);
			}

			else if(strncmp(pch,"cannyhough",8)==0){
				utility::cannyhough(I, I2);
			}

			else if(strncmp(pch,"smoothedge",8)==0){
				utility::smoothedge(I, I2);
			}

/*---------------------------------------------------------------------------------------------------------------**
***									PART 3: COMBINING OPERATIONS WITH OPENCV									***
*---------------------------------------------------------------------------------------------------------------**/
			else if(strncmp(pch,"otsu",8)==0){
				utility::cv_otsu(I, I2);
			}
			
			else if(strncmp(pch,"otsuhisto",8)==0){
				utility::otsuhisto(I, I2);
			}

			else if(strncmp(pch,"otsugau",8)==0){
				utility::otsugau(I, I2);
			}

			else if(strncmp(pch,"combine",8)==0){
				utility::combine(I, I2);
			}

			// else if(strncmp(pch,"subtract",8)==0){
			// 	// I3 = strtok(NULL, " ");
			// 	utility::subtract(I, I_2, I2);
			// }

/*---------------------------------------------------------------------------------------------------------------**
***									PART 4: ADVANCED OPERATIONS (EXTRA CREDIT)									***
*---------------------------------------------------------------------------------------------------------------**/
			else if(strncmp(pch,"QRcode",8)==0){
				utility::QRcode(I, I2);
			}

			else if(strncmp(pch,"QRcodepre",8)==0){
				utility::QRcodepre(I, I2);
			}

			else {
				printf("No function: %s\n", pch);
				continue;
			}
			
			cv::imwrite(outfile, I2);
		}
		else {
			src.read(infile);
			if (strncmp(pch,"add",3)==0) {
				pch = strtok(NULL, " ");
				utility::addGrey(src,tgt,atoi(pch));
			}

			else if (strncmp(pch,"binarize",8)==0) {
				pch = strtok(NULL, " ");
				utility::binarize(src,tgt,atoi(pch));
			}

			else if (strncmp(pch,"scale",5)==0) {
				pch = strtok(NULL, " ");
				utility::scale(src,tgt,atof(pch));
			}

			else {
				printf("No function: %s\n", pch);
				continue;
			}
			
			tgt.save(outfile);
		}
	}
	fclose(fp);
	return 0;
}

