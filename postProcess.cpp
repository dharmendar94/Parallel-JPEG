#include <iostream>
#include<stdlib.h>
#include<fstream>

inline void addHeader(FILE** file, unsigned int numRows, int numCols){
	FILE* h = fopen("H.bin", "rb");
	if (h == NULL){
		printf("h.bin error");
		exit(-1);
	}

	unsigned int dim1, dim2;
	unsigned char c;
	for (int i = 0; i < 623; i++){
		if (i == 163){
			dim1 = numRows >> 8;
			fputc((char)dim1,*file);
			numRows = numRows & 255;
			fputc((char)numRows, *file);

			dim1 = numCols >> 8;
			fputc((char)dim1, *file);
			numCols = numCols & 255;
			fputc((char)numCols, *file);

			i += 4;
			fgetc(h); fgetc(h); fgetc(h); fgetc(h);

		}
		
		c = fgetc(h);
		fputc(c, *file);
		
	}
	fclose(h);
}

int postProcess(std::string inter, std::string outputFIle, unsigned int numRows, unsigned int numCols)
{
	FILE *fin = fopen(inter.c_str(),"rb");
	if (fin == NULL){
		printf("Cannot open comp.bin");
		system("pause");
		exit(-1);
	}

	FILE *fout = fopen(outputFIle.c_str(), "wb");
	if (fout == NULL){
		printf("Cannot open comp.bin");
		system("pause");
		exit(-1);
	}
	addHeader(&fout, numRows, numCols);
	unsigned char byte, stuff;
	unsigned int test = 0;
	stuff = '0' - 48;
	char c;
	c = fgetc(fin);

	while (c != EOF){
		byte = byte & 0;
		byte = c - 48;
		c = fgetc(fin);
		for (int i = 0; i < 7 && c != EOF; i++){

			byte = byte << 1;
			byte = byte | (c - 48);
			c = fgetc(fin);
		}
		test = byte;
		fwrite(&byte, 1, 1, fout);
		if (byte == 255)
			fwrite(&stuff, 1, 1, fout);
	}

	fputc((char)255, fout);
	fputc((char)217, fout);
	fclose(fin);
	fclose(fout);
	return 0;
}
