#define _CRT_SECURE_NO_WARNINGS
//数组：一组相同类型元素的集合
//数组定义 
//   int arr[10] = { 1,2,3,4,5 }; 定义一个整型数组，最多放10个元素
#include <stdio.h>
int main_06()
{
	/*int a = 1;
	int b = 2;
	int c = 3;*/
	int arr[10] = { 1,2,3,4,5,6,7,8,9,10 };//定义一个存放10个整型数字的数组
	//printf("%d\n", arr[4]);  //用下标的方式访问元素
	int i = 0;
	while (i < 10)
	{
		printf("%d", arr[i]);
		i++;
	}

	/*char ch[20];
	float arr2[5];*/


	return 0;
}