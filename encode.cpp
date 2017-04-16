#include<stdio.h>
#include<math.h>
int returnSize(int val){
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
