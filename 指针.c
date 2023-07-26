#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int ma;;; in()
{
	int a = 10;
   int*	p = &a;//取地址a
	//有一种变量用来存放地址-指针变量
	printf("%p\n", &a);
	printf("%p\n", p);
	*p = 20;//解引用操作符
	printf("a = %d\n", a);
	char ch = 'w';
	char* pc = &ch;
	*pc = 'a';
	printf("%c\n", ch);


	

	//指针变量大小
	//printf("%d\n", sizeof(char *));
	//结论：指针大小在32位平台4个字节  64位平台八个字节



	return 0;
}
//如何产生地址
//32位 32根地址线/数据线  一旦通电会有正电和负电 1/0 
// 无非是0000000000
// 000000000000001
// 000000000000010
// ......
// 有2^32个序列   
//
//
//
//
//