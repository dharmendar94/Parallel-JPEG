
//This file declares the global variable used in device and host

//device 
extern int *d_Y;
extern  int *d_Cb;
extern  int *d_Cr;
extern int *d_DCTY;
extern  int2 *d_DCTCbCr;
extern uchar3* d_inputImage;


//host 
extern uchar3* h_inputImage;
extern int *h_DCTY;
extern int2 *h_DCTCbCr;

//huffman tables
extern char* HuffmanDC_Luma[12];
extern char* HuffmanAC_Luma[16][11];
extern char * HuffmanDC_Chroma[12];
extern char* HuffmanAC_Chroma[16][11];
extern char returnSize(int);

//////functions
void preProcess(uchar3 **h_inputImage, uchar3 **d_inputImage, unsigned int numRows, unsigned int numCols);
int postProcess(std::string inter, std::string outputFIle);

void RGBToYCbCrHelper(uchar3 *d_inputImage, int** d_Y, int** d_Cb, int** d_Cr, unsigned int numRows, unsigned int numCols);

void FDCTHelper(int *d_Y, int *d_Cb, int *d_Cr, int **d_DCTY, int2 **d_DCTCbCr, int**h_DCTY, int2 **h_DCTCbCr,
	unsigned int Y_numRows, unsigned int Y_numCols);

void compressImage(int *DCTY, int2 *DCTCbCr, FILE** outputFile, unsigned int numRows, unsigned int numCols);

//handle errors
extern void checkErrors(cudaError_t err, char* string);


