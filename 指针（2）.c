#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main()
{
	//字符指针
	//char ch = 'w';
 //  char* pc = &ch;
 //  char* p = "abcdef";//放的是首元素地址
 //  printf("%c\n", "abcdef"[3]);



   //指针数组
   //字符数组 - 存放字符的数组、 指针数组 - 存放指针的数组，存放在数组中的元素都是指针类型
//
   //可以使用指针数组模拟一个二维数组




  // int arr1[] = { 1,2,3,4,5 };
  // int arr2[] = { 2,3,4,5,6 };
  // int arr3[] = { 3,4,5,6,7 };

  //int* arr[] = { arr1,arr2,arr3 };
  ////指针数组
  //int i = 0;
  //for (i = 0; i < 3; i++)
  //{
	 // int j = 0;
	 // for (j = 0; j < 5; j++)
	 // {
		//  printf("%d", arr[i][j]);
	 // }
	 // printf("\n");
  //


	//数组指针 - --是数组，存放指针的数组
	 //指针数组
	//数组指针 - 是指针
	//字符指针 - 是指向字符的指针
	//整形指针 - 是指向整形的指针

	//数组指针--是执行那个数组的指针

	//int arr[10] = { 0 };

	//int(*p)[10] = &arr;//p是用来存放数组的地址的 p就是数组指针

	//char* arr2[5];
	//char* (*pc)[5] = &arr2;





	//函数指针
	//指向函数的指针
	//&函数名就可以
	//函数名就是函数的地址
//	int (*pf1)(int, int) = Add;


	(*(void (*)())0)();

	void(*singnal(int, void(*)(int)))(int);












	return 0;
}