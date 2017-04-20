
/*#include"CUDA_headers.h"
#include"globals.h"

__constant__  float coef[8] = { 0.707, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

__global__ void IDCT_2D(uchar *d_IDCT, float * d_DCT, unsigned int numRows, unsigned int numCols){

	int tx = threadIdx.x;								    int ty = threadIdx.y;
	int bx = blockIdx.x;									int by = blockIdx.y;
	int pos_x = tx + blockDim.x  *  bx;						int pos_y = ty + blockDim.y  *  by;
	float sum;

	__shared__ float M[8][8];
	__shared__ float block[8][8];

	M[ty][tx] = 0;
	__syncthreads();

	if (pos_x >= numCols || pos_y >= numRows)
		return;

	M[ty][tx] = d_DCT[pos_x + pos_y  *  numCols];

	__syncthreads();



	////TRANFORMATION/////////////////

	sum = 0.0f;
	for (int v = 0; v < 8; v++){
		for (int u = 0; u < 8; u++){
			sum += coef[u] * coef[v] * M[v][u] *
				cosVal[u][tx] * cosVal[v][ty];
		}
	}
	sum /= 4.0f;

	sum = round(sum);

	sum += 128.0f;
	__syncthreads();



	block[ty][tx] = sum;
	__syncthreads();

	if (block[ty][tx] < 150)
		block[ty][tx] = 0;
	d_IDCT[pos_x + pos_y * numCols] = (int)block[ty][tx];

}



void IDCT_2DHelper(uchar *d_IDCT, float * d_DCT, unsigned int numRows, unsigned int numCols){
	cudaError_t err;
	const dim3 blockSize1(8, 8, 1);
	const dim3 gridSize1(numCols / 8 + 1, numRows / 8 + 1, 1);
	IDCT_2D << <gridSize1, blockSize1 >> >(d_IDCT, d_DCT, numRows, numCols);
	err = (cudaDeviceSynchronize());
	if (err != cudaSuccess){
		printf("IDCT_2D sync failed  %s", cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
	err = (cudaGetLastError());
	if (err != cudaSuccess){
		printf("IDCT_2D failed  %s", cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
}
*/