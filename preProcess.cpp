#include"CUDA_headers.h"
#include"errorCheck.h"

//define the extern variables

void preProcess(uchar3 **h_inputImage, uchar3 **d_inputImage, unsigned int numRows, unsigned int numCols){
	cudaError_t err;
	unsigned int numPixels = numRows * numCols;

	//allocate memory for input image
	err = cudaMalloc((void**)d_inputImage, numPixels * sizeof(uchar3));
	checkErrors(err, "cudaMalloc(d_inputImage)");

	err = cudaMemcpy(*d_inputImage, *h_inputImage, numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);
	checkErrors(err, "cudaMemcpy(d_inputImage, h_inputImage)");

}