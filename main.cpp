#include"OpenCV_headers.h"
#include"CUDA_headers.h"
#include"globals.h"
#include"calcTime.h"
#include<stdio.h>
#include<fstream>

cv::Mat inputImage;
cv::Mat outputImage;

std::string inputFile = "G:\\sampleImages\\bmp3.bmp";
std::string outputFile = "bmp3.txt";

//get number of rows and columns of input image
unsigned int numRows(){ return inputImage.rows; }
unsigned int numCols(){ return inputImage.cols; }

int main(){
	//freopen("Y_DCT.txt","w",stdout);
	
	checkErrors(cudaFree(0),"cudaFree(0)");
	//StartCounter();

	//Load inputImage;
	inputImage = cv::imread(inputFile.c_str() , CV_LOAD_IMAGE_COLOR);
	if (inputImage.empty()){
		printf("Failed to load Image.\n");
		return 0;
	}

	//Set a pointer to inputImage in host memory
	h_inputImage = (uchar3*)inputImage.ptr<unsigned int>(0);
	
	//This function allocates memory for input image variable d_inputImage 
	//on deivce.
	preProcess(&h_inputImage, &d_inputImage, numRows(), numCols());

	//Convert RGB colour space to YCbCr and downsample the Cb and Cr channels
	RGBToYCbCrHelper(d_inputImage, &d_Y, &d_Cb, &d_Cr, numRows(), numCols());

	//Call kernel to calculate DCT values for Y, Cb, Cr channels.
	FDCTHelper(d_Y, d_Cb, d_Cr, &d_DCTY, &d_DCTCbCr, &h_DCTY, &h_DCTCbCr, numRows(), numCols());

	//Open output file
	FILE *outputFIle = fopen("bmp3.txt","wb");
	//Entropy encoding of DCT values.
	compressImage(h_DCTY, h_DCTCbCr, outputFile, numRows(), numCols());

	system("pause");
	cudaFree(d_DCTY);
	cudaFree(d_DCTCbCr);
	return 0;
}

