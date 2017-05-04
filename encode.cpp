#include<stdio.h>
#include<math.h>
#include<string.h>
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


void toBinary(int value, char *code, int size, int tlen){
	
	if (value < 0){
		int absVal = abs(value);
		
		for (int i = tlen - 1; i > tlen - size - 1; i--){
			if (absVal % 2 == 0)
				code[i] = '1';
			else
				code[i] = '0';
			absVal /= 2;
		}
	}
	else{
		
		for (int i = tlen - 1; i > tlen - size - 1; i--){
			if (value % 2 == 0)
				code[i] = '0';
			else
				code[i] = '1';
			value /= 2;
		}
	}
	code[tlen] = '\0';
	return;
}
void encodeBlock(int RL[][2], int rlelen, FILE **outputFile, int id){
	int size, tlen;
	char code[32];
	size = returnSize(RL[0][1]);
	
	if (id){
		//Huffman encode CbCr DC size 
		tlen = lengthDC_chroma[size];
		strcpy(code, HuffmanDC_chroma[size]);
		toBinary(RL[0][1], code, size, tlen);

		//write encoded information to file.
		fwrite(code, sizeof(char), tlen, *outputFile);

		//Huffman encode Cb, Cr AC size values, AC values to binary.
		for (int i = 0; i < rlelen; i++){
			size = returnSize(RL[i][1]);

			tlen = lengthAC_chroma[RL[i][0]][size];
			strcpy(code, HuffmanAC_chroma[RL[i][0]][size]);

			toBinary(RL[i][1], code, size, tlen);
			fwrite(code, sizeof(char), tlen, *outputFile);
		}
	}
	else{
		//Huffman encode Y DC size value
		strcpy(code,HuffmanDC_luma[size]);
		tlen = lengthDC_luma[size];
		toBinary(RL[0][1], code, size, tlen);
		
		//write encoded information to file.
		fwrite(code, sizeof(char), tlen, *outputFile);
		
		//Huffman encode Y AC size values, AC values to binary.
		for (int i = 0; i < rlelen; i++){
			size = returnSize(RL[i][1]);

			tlen = lengthAC_luma[RL[i][0]][size];
			strcpy(code, HuffmanAC_luma[RL[i][0]][size]);

			toBinary(RL[i][1], code, size, tlen);
			fwrite(code, sizeof(char), tlen, *outputFile);

		}
	}
	
}

int dct1[8][8];
int dct2[8][8];

int runLengthEncode(int *zz, int RL[][2]){
	int i, j, count;
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


int compress(int dct[8][8], int dcVal, FILE **outputFile,int id){
	
	int rlen=0, RL[64][2];
	
	int newDC = dct[0][0];
	dct[0][0] -= dcVal;

	//Zig Zag
	int ZZ[64];
	for (int i = 0, k = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			ZZ[z[i][j]] = dct[i][j];

	rlen = runLengthEncode(ZZ, RL);

	encodeBlock(RL, rlen, outputFile, id);

	return newDC;
}


void compressImage(int *DCTY, int2 *DCTCbCr, FILE** outputFile, unsigned int numRows, unsigned int numCols){

	unsigned int CbCr_numRows = (numRows % 2 == 0) ? (numRows / 2) : (numRows / 2 + 1);
	unsigned int CbCr_numCols = (numCols % 2 == 0) ? (numCols / 2) : (numCols / 2 + 1);

	int YDC = 0;
	int CbDC = 0;
	int CrDC = 0;
	int Yblocks_y = (numRows + 7) / 8;
	int Yblocks_x = (numCols + 7) / 8;
	int Crblocks_y = (CbCr_numRows + 7) / 8;
	int Crblocks_x = (CbCr_numCols + 7) / 8;

	YDC = CbDC = CrDC = 0;
	for (int by = 0; by < Yblocks_y; by+=2){
		
		for (int bx = 0; bx < Yblocks_x; bx+=2){
			
			//encode Y
			for (int r = 0; r < 8; r++){
				for (int c = 0; c < 8; c++){
					if ((by * 8 + r) >= numRows || (bx * 8 + c) >= numCols){
						dct1[r][c] = 0;
						dct2[r][c] = 0;
					}
					else{
						dct1[r][c] = DCTY[(bx * 8 + c) + (by * 8 + r)*numCols];
						dct2[r][c] = DCTY[((bx + 1) * 8 + c) + (by * 8 + r)*numCols];

					}
				}
			}
			YDC = compress(dct1, YDC, outputFile,0);
			YDC = compress(dct2, YDC, outputFile,0);
			 
			for (int r = 0; r < 8; r++){
				for (int c = 0; c < 8; c++){
					if ((by * 8 + r) >= numRows || (bx * 8 + c )>= numCols){
						dct1[r][c] = 0;
						dct2[r][c] = 0;
					}
					else{
						dct1[r][c] = DCTY[(bx * 8 + c) + ((by+1) * 8 + r)*numCols];
						dct2[r][c] = DCTY[((bx+1) * 8 + c) + ((by+1) * 8 + r)*numCols];
						
					}
				}
			}

			YDC = compress(dct1, YDC, outputFile, 0);
			YDC = compress(dct2, YDC, outputFile, 0);

			//encode Cb,Cr
			for (int r = 0; r < 8; r++){
				for (int c = 0; c < 8; c++){
					if ((by * 8 + r) >= numRows/2 || (bx * 8 + c) >= numCols/2){
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
