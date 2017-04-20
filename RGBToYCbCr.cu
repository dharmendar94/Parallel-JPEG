/*
This file contains the code for converting RGB to YCbCr colour space and 
downsampling the Cb and Cr channels.
*/


#include"CUDA_headers.h"
#include"calcTime.h"
#include"errorCheck.h"
#include<math.h>

__global__
void RGBToYCbCr(uchar3 *d_inputImage,int* d_Y, int* d_Cb, int* d_Cr, 
				unsigned int numRows, unsigned int numCols, unsigned int rows, unsigned int cols){

	int idx = threadIdx.x;		int idy = threadIdx.y;

	int tx = idx + blockIdx.x * blockDim.x;
	int ty = idy + blockIdx.y * blockDim.y;
	int index = tx + ty * numCols;

	//shared memory for Cb and Cr channels
	__shared__ int _Cb[32][32];
	__shared__ int _Cr[32][32];

	_Cb[idy][idx] = 100;
	_Cr[idy][idx] = 100;

	if (tx >= numCols || ty >= numRows)
		return;



	float Cb_avg,Cr_avg;
	int x = d_inputImage[index].x;
	int y = d_inputImage[index].y;
	int z = d_inputImage[index].z;	
	
	d_Y[index] =	(int)round(0.299f * z + 0.587f * y + 0.114f * x);

	_Cb[idy][idx] = (int)round(128 - 0.1687f * z - 0.3313f * y + 0.5f * x);

	_Cr[idy][idx] = (int)round(128 + 0.5f * z - 0.4187f * y - 0.0813f * x);

	//if (ty == 0)
	//	printf("%d __ %d\n", tx, _Cb[idy][idx]);

	//make shure that all threads calculate Y, Cb and Cr values before downsampling
	__syncthreads();

	/////**Downsample the Cb and Cr channels***///////////
	if (ty % 2 == 0 && tx % 2 == 0){

		Cb_avg = roundf((float)(_Cb[idy][idx] + _Cb[idy][idx + 1] + _Cb[idy + 1][idx] + _Cb[idy + 1][idx + 1]) / 4.0f);
		
		Cr_avg = roundf((float)(_Cr[idy][idx] + _Cr[idy][idx + 1] + _Cr[idy + 1][idx] + _Cr[idy + 1][idx + 1]) / 4.0f);
		
		d_Cb[(tx / 2) + (ty / 2) * cols] = (int)roundf(Cb_avg);
		d_Cr[(tx / 2) + (ty / 2) * cols] = (int)roundf(Cr_avg);

		//if (ty == 0)
			//printf("(%d _ %d, %d)  %d _ %d _ %d _ %d -> %d \n",tx, idy, idx, _Cb[idy][idx], _Cb[idy][idx + 1], _Cb[idy + 1][idx], _Cb[idy + 1][idx + 1],d_Cb[(tx / 2) + (ty / 2) * cols]);
	}
	

}

void RGBToYCbCrHelper(uchar3 *d_inputImage, int** d_Y, int** d_Cb, int** d_Cr, unsigned int numRows, unsigned int numCols){

	cudaError_t err;
	
	//rows and colums for the downsampled channels
	unsigned int rows = (numRows % 2 == 0) ? (numRows / 2) : (numRows / 2 + 1);
	unsigned int cols = (numCols % 2 == 0) ? (numCols / 2) : (numCols / 2 + 1);

	//allocate memory for YCbCr
	err = cudaMalloc((void**)d_Y, numRows * numCols * sizeof(int));
	checkErrors(err, "cudaMalloc(Y)");

	err = cudaMalloc((void**)d_Cb, rows * cols * sizeof(int));
	checkErrors(err, "cudaMalloc(_Cb)");
	
	err = cudaMalloc((void**)d_Cr, rows * cols * sizeof(int));
	checkErrors(err, "cudaMalloc(_Cr)");

	const dim3 blockSize(32, 32, 1);
	const dim3 gridSize(numCols / 32 + 1, numRows / 32 + 1, 1);
	
	//StartCounter();

	RGBToYCbCr <<< gridSize, blockSize >> > (d_inputImage, *d_Y, *d_Cb, *d_Cr, numRows, numCols, rows, cols);
	
	err = cudaDeviceSynchronize();
	checkErrors(err, "RGBToYCbCr : cudaDeviceSynchronize() ");
	err = cudaGetLastError();
	checkErrors(err, "RGBToYCbCr<<<,>>>");

	//Free input image memory on device
	err = cudaFree(d_inputImage);
	//printf("Time : %f\n", GetCounter());
}