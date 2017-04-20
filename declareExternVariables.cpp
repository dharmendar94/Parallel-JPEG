#include<cuda_runtime.h>

//device 
int *d_Y;
int *d_Cb;
int *d_Cr;
int *d_DCTY;
int2 *d_DCTCbCr;

uchar3* d_inputImage;

//host 
uchar3* h_inputImage;
int *h_DCTY;
int2 *h_DCTCbCr;