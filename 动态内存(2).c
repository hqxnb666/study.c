#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void GetMemory(char* p)
{
	p = (char*)malloc(100);
}
void Test(void)
{
	char* str = NULL;
	GetMemory(str);
	strcpy(str, "hello word");
	printf(str);
}
//   1,�����NULL�����ò��� �������
//  �ڴ�й©


void GetMemory(char** p)
{
	*p = (char*)malloc(100);
}
void Test(void)
{
	char* str = NULL;
	GetMemory(&str);
	strcpy(str, "hello word");
	printf(str);
	free(str);
	str = NULL;
}
int main()
{

	return 0;
}