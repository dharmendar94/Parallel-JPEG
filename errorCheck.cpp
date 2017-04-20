#include"CUDA_headers.h"

void checkErrors(cudaError_t err, char* string){
	if (err != cudaSuccess){
		printf("\n%s\nError: %s\n", string, cudaGetErrorString(err));
		system("pause");
	}
}
