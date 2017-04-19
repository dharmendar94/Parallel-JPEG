#include<stdio.h>
#include<math.h>
#include<cuda_runtime.h>
#include"huffmanTables.h"

inline int returnSize(int val){
	int absVal = abs(val);
	if (absVal == 0)
		return 0;
	else if (absVal <= 1)
		return 1;
	else if (absVal <= 3)
		return 2;
	else if (absVal <= 7)
		return 3;
	else if (absVal <= 15)
		return 4;
	else if (absVal <= 31)
		return 5;
	else if (absVal <= 63)
		return 6;
	else if (absVal <= 127)
		return 7;
	else if (absVal <= 225)
		return 8;
	else if (absVal <= 511)
		return 9;
	else if (absVal <= 1023)
		return 10;
	else if (absVal <= 2047)
		return 11;
}


void encode(int RL[][2], int len, FILE *outputFile, int id){
	int size;
	char code[32];
	size = returnSize(RL[0][1]);
	

	for (int i = 0; i < len; i++){
		size = returnSize(RL[i][1]);

	}
}

int dct1[8][8];
int dct2[8][8];

int runLengthEncode(int *zz, int RL[][2]){
	int i, j, eob, count;
	int rlen = 0;
	i = 63;
	if (zz[63] == 0){
		while (i >= 0 && zz[i] == 0)i--;
		if (i < 0){
			RL[rlen][0] = 0;
			RL[rlen++][1] = 0;
		}
		else{
			count = 0;
			for (j = 0; j <= i; j++){
				if (zz[j] == 0){
					++count;
				}
				else{
					RL[rlen][0] = count;
					RL[rlen++][1] = zz[j];
					count = 0;

				}
				if (count > 15){
					RL[rlen][0] = 15;
					RL[rlen++][1] = 0;
					count = 0;
				}
			}
			RL[rlen][0] = 0;
			RL[rlen++][1] = 0;
		}
	}
	else{
		count = 0;
		for (j = 0; j <= i; j++){
			if (zz[j] == 0){
				++count;
			}
			else{
				RL[rlen][0] = count;
				RL[rlen++][1] = zz[j];
				count = 0;

			}
			if (count > 15){
				RL[rlen][0] = 15;
				RL[rlen++][1] = 0;
				count = 0;
			}
		}
	}
	return (rlen);
}


int compress(int dct[8][8], int dcVal, FILE *outputFile,int id){
	
	int rlen=0, RL[64][2];
	
	int newDC = dct[0][0];
	dct[0][0] -= dcVal;

	//Zig Zag
	int ZZ[64];
	for (int i = 0, k = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			ZZ[z[i][j]] = dct[i][j];

	rlen = runLengthEncode(ZZ, RL);

	encode(RL, rlen, outputFile, id);

	return newDC;
}


void compressImage(int *DCTY, int2 *DCTCbCr, FILE* outputFile, unsigned int numRows, unsigned int numCols){

	unsigned int CbCr_numRows = (numRows % 2 == 0) ? (numRows / 2) : (numRows / 2 + 1);
	unsigned int CbCr_numCols = (numCols % 2 == 0) ? (numCols / 2) : (numCols / 2 + 1);

	int YDC = 0;
	int CbDC = 0;
	int CrDC = 0;
	int Yblocks_y = (numRows + 7) / 8;
	int Yblocks_x = (numCols + 7) / 8;
	int Crblocks_y = (CbCr_numRows + 7) / 8;
	int Crblocks_x = (CbCr_numCols + 7) / 8;

	for (int by = 0; by < Yblocks_y; by+=2){
		YDC = CbDC = CrDC = 0;
		for (int bx = 0; bx < Yblocks_x; bx+=2){
			
			//encode Y
			for (int r = 0; r < 8; r++){
				for (int c = 0; c < 8; c++){
					if (by * 8 + r >= Yblocks_y || bx * 8 + c >= Yblocks_x){
						dct1[r][c] = 0;
						dct2[r][c] = 0;
					}
					else{
						dct1[r][c] = DCTY[(bx * 8 + c) + (by * 8 + r)*numCols];
						dct2[r][c] = DCTY[((bx + 1) * 8 + c) + (by * 8 + r)*numCols];

					}
				}
			}
			YDC = compress(dct1,YDC,0);
			YDC = compress(dct2, YDC,0);
			 
			for (int r = 0; r < 8; r++){
				for (int c = 0; c < 8; c++){
					if (by * 8 + r >= Yblocks_y || bx * 8 + c >= Yblocks_x){
						dct1[r][c] = 0;
						dct2[r][c] = 0;
					}
					else{
						dct1[r][c] = DCTY[((bx + 1) * 8 + c) + (by * 8 + r)*numCols];
						dct2[r][c] = DCTY[(bx * 8 + c) + (by * 8 + r)*numCols];
						
					}
				}
			}

			YDC = compress(dct1, YDC, outputFile, 0);
			YDC = compress(dct2, YDC, outputFile, 0);

			//encode Cb,Cr
			for (int r = 0; r < 8; r++){
				for (int c = 0; c < 8; c++){
					if (by * 8 + r >= (Yblocks_y/2) || bx * 8 + c >= (Yblocks_x/2)){
						dct1[r][c] = 0;
						dct2[r][c] = 0;
					}
					else{
						dct1[r][c] = DCTCbCr[((bx/2) * 8 + c) + ((by/2) * 8 + r)*numCols].x;
						dct2[r][c] = DCTCbCr[((bx/2) * 8 + c) + ((by/2) * 8 + r)*numCols].y;
					}
				}
			}

			CbDC = compress(dct1, CbDC, outputFile, 1);
			CrDC = compress(dct2, CrDC, outputFile, 1);
		}
	}

}
