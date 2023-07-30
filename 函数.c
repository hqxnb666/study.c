#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
 //int Add(int x, int y)
//{
	//int z = 0;
	//z = x + y;
	//return z;
//}
int masin()
{
	/*int a = 10;
	int b = 20;
  int sum =	Add(a, b);
  printf("%d\n", sum);*/

  //C语言中函数的分类
  // 1.库函数  Io函数  字符串操作函数  字符操作函数   内存操作函数   时间/日期函数  数学函数  其他库函数
  // 2.自定义函数



    //strcpy 字符串拷贝
    // char * strcpy (cahr * destination, const cahr * source);
	//char arr1[] = "bit";
	//char arr2[] = "######";
	//strcpy(arr2, arr1);
	//printf("%s\n", arr2);


	//memset  void * memset (void *ptr, int value, size_t num );
	char arr[] = "hello word";
	memset(arr, '*', 5);
	printf("%s\n", arr);
	return 0;
}