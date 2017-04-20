/*	This file includes kernels for calculating FDCT values in the encoding process of the input image file.
	Two separate kernels are defined for Y and Cb, Cr channels.
	The kernels first shift the channel values and then calculate FDCT.
*/

#include"CUDA_headers.h"
#include"values.cuh"
#include"errorCheck.h"
#include"calcTime.h"

/*	This kernel calculates DCT values for Y channel only.
	Because:
	The number of rows and columns of Y channel are double as compare to of Cb and Cr channels.
	Using only a single kernel introduces more branching in it resulting in lower performance.   
*/


__global__ 
void YChannelFDCT_2D(int *d_Y, int *d_DCTY, unsigned int numRows, unsigned int numCols){

	int tx = threadIdx.x;								    int ty = threadIdx.y;
	int bx = blockIdx.x;									int by = blockIdx.y;
	int pos_x = tx + blockDim.x  *  bx;						int pos_y = ty + blockDim.y  *  by;

	float sum;
	__shared__ float M[8][8];
	__shared__ float block[8][8];

	M[ty][tx] = 0.0f;

	__syncthreads();

	if (pos_x >= numCols || pos_y >= numRows)
		return;

	M[ty][tx] = (float)d_Y[pos_x + pos_y  *  numCols] - 128.0f;
	 
	__syncthreads();

	//////transform rows (r = constant)//////////////
	sum = 0;
	for (int c = 0; c < 8; c++){
		sum += M[ty][c] * cosVal[tx][c];
	}
	if (tx == 0) sum *= (1.0 / sqrtf(2.0));
	block[ty][tx] = sum / 2.0;
	__syncthreads();

	///transform columns (c = constant)/////////////
	sum = 0;
	for (int r = 0; r < 8; r++){
		sum += block[r][tx] * cosVal[ty][r];
	}
	if (ty == 0)sum *= (1.0 / sqrtf(2.0));
	M[ty][tx] = sum / 2.0f;

	__syncthreads();

	d_DCTY[pos_x + pos_y * numCols] = (int)(roundf(M[ty][tx] / (float)Y_quantMatrix[ty][tx]));

}


//A separate kernel to calculate DCT values for Cb and Cr channles
__global__ void CbCrChannelFDCT_2D(int *d_Cb, int *d_Cr, int2 *d_DCTCbCr,
									unsigned int numRows, unsigned int numCols){

	int tx = threadIdx.x;								    int ty = threadIdx.y;
	int bx = blockIdx.x;									int by = blockIdx.y;
	int pos_x = tx + blockDim.x  *  bx;						int pos_y = ty + blockDim.y  *  by;

	float sum_Cb, sum_Cr;

	__shared__ float M_Cb[8][8];
	__shared__ float block_Cb[8][8];

	__shared__ float M_Cr[8][8];
	__shared__ float block_Cr[8][8];

	M_Cb[ty][tx] = 0.0f;
	M_Cr[ty][tx] = 0.0f;

	__syncthreads();
	
	if (pos_x >= numCols || pos_y >= numRows)
		return;

	//Shifting 
	M_Cb[ty][tx] = (float)d_Cb[pos_x + pos_y  *  numCols] - 128.0f;
	M_Cr[ty][tx] = (float)d_Cr[pos_x + pos_y  *  numCols] - 128.0f;
	 
	__syncthreads();

	//////transform rows (r = constant)//////////////
	sum_Cb = 0.0f;
	sum_Cr = 0.0f;
	for (int c = 0; c < 8; c++){
		sum_Cb += M_Cb[ty][c] * cosVal[tx][c];
		sum_Cr += M_Cr[ty][c] * cosVal[tx][c];
	}
	
	if (tx == 0){
		sum_Cb *= (1.0 / sqrtf(2.0));
		sum_Cr *= (1.0 / sqrtf(2.0));
	}
	block_Cb[ty][tx] = sum_Cb / 2.0f;
	block_Cr[ty][tx] = sum_Cr / 2.0f;
	
	__syncthreads();

	///transform columns (c = constant)/////////////
	sum_Cb = 0;
	sum_Cr = 0;
	for (int r = 0; r < 8; r++){
		sum_Cb += block_Cb[r][tx] * cosVal[ty][r];
		sum_Cr += block_Cr[r][tx] * cosVal[ty][r];
	}

	if (ty == 0){
		sum_Cb *= (1.0 / sqrtf(2.0));
		sum_Cr *= (1.0 / sqrtf(2.0));
	}

	M_Cb[ty][tx] = sum_Cb / 2.0;
	M_Cr[ty][tx] = sum_Cr / 2.0;

	__syncthreads();
	
	//Perform quantization 

	d_DCTCbCr[pos_x + pos_y * numCols].x = (int)(roundf(M_Cb[ty][tx] / CbCr_quantMatrix[ty][tx]));
	d_DCTCbCr[pos_x + pos_y * numCols].y = (int)(roundf(M_Cr[ty][tx] / CbCr_quantMatrix[ty][tx]));

}


void FDCTHelper(int *d_Y, int *d_Cb, int *d_Cr, int **d_DCTY, int2 **d_DCTCbCr, int **h_DCTY, int2 **h_DCTCbCr,
				unsigned int Y_numRows, unsigned int Y_numCols){

	cudaError_t err;
	
	//***********************************************************************************************//
	//////****************************** Y channel section ***************************/////////////////
	//***********************************************************************************************//


	//Allocate memory for FDCT values for Y channel.
	err = cudaMalloc((void**)d_DCTY, Y_numRows * Y_numCols * sizeof(int));
	checkErrors(err, "cudaMalloc(d_DCTY)");

	const dim3 Y_blockSize(8, 8, 1);
	const dim3 Y_gridSize(Y_numCols / 8 + 1, Y_numRows / 8 + 1, 1);

	YChannelFDCT_2D << <Y_gridSize, Y_blockSize >> >(d_Y, *d_DCTY, Y_numRows, Y_numCols);

	err = cudaDeviceSynchronize();
	checkErrors(err, "YChannelFDCT_2D : cudaDeviceSynchronize");

	err = cudaGetLastError();
	checkErrors(err, "YChannelFDCT_2D<<< , >>>");

	//Free Y channel memory in device.
	err = cudaFree(d_Y);
	checkErrors(err, "cudaFree(d_Y)");


	//***********************************************************************************************//
	///////////****************** Cb, Cr channel section ****************************//////////////////
	//***********************************************************************************************//

	unsigned int CbCr_numRows = (Y_numRows % 2 == 0) ? (Y_numRows / 2) : (Y_numRows / 2 + 1);
	unsigned int CbCr_numCols = (Y_numCols % 2 == 0) ? (Y_numCols / 2) : (Y_numCols / 2 + 1);

	//Allocate memory for FDCT values of Cb and Cr channels.
	err = cudaMalloc((void**)d_DCTCbCr, CbCr_numRows * CbCr_numCols * sizeof(int2));
	checkErrors(err, "cudaMalloc(d_DCTCbCr)");

	const dim3 CbCr_blockSize(8,8,1);
	const dim3 CbCr_gridSize(CbCr_numCols / 8 + 1, CbCr_numRows / 8 + 1);

	CbCrChannelFDCT_2D << <CbCr_gridSize, CbCr_blockSize >> >(d_Cb, d_Cr, *d_DCTCbCr, CbCr_numRows, CbCr_numCols);
	
	err = cudaDeviceSynchronize();
	checkErrors(err, "CbCrChannelFDCT_2D : cudaDeviceSynchronize");

	err = cudaGetLastError();
	checkErrors(err, "CbCrChannelFDCT_2D<<< , >>>");

	//Free Cb and Cr channels memory in device
	err = cudaFree(d_Cb);
	checkErrors(err, "cudaFree(d_Cb)");

	err = cudaFree(d_Cr);
	checkErrors(err, "cudaFree(d_Cr)");
}
