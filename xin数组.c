#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int mn()
{
	//数组存放元素不能为0 存放多个数据 类型相同

	//基本语法如下
	//type arr_name[常量值]
	// type 可以是  char short int ....
	//arr_name 有意义就行
	//[]表示数组大小


	//int num[5] = { 1,2,3,4,5 };//完全初始化
	//int num2[5] = { 1,2,3 };//不完全 剩余为0
	//int num3 [] = {1,2,3};//如果数组初始化了 是可以省略数组的
	
	//数组是有类型的 arr1 的类型是  int [10] ch数组的类型是 char [5]



	//一维数组

	//下标引用操作符
	//arr[6]


	/*int num1[10] = { 1,2,3,4,5,6,7,8,9,10 };
	int i = 0;
	for (i = 0; i < 10; i++)
		printf("&arr[%d] = %p\n", &num1[i]);*/ // %p打印地址
	 //一维数组内存中是连续存放的 随着下标的增长 地址由低到高变化


	//int arr[10] = { 1,2,3 };
	//printf("%d\n", sizeof(arr));//  sizeof 计算的是数组大小 单位是字节
	//printf("%d\n", sizeof(arr[0]));
	//printf("%d\n", sizeof(arr) / sizeof(arr[0]));

	//int sz = sizeof(arr) / sizeof(arr[0]);




		
	

	// 二维数组
	// tyoe arr_name[常量值1][常量值2]
	// 
	// int arr[3][5]
	// double data[2][8]
	// 3表示数组有三行 5表示每一行五个元素

	//int arr[3][5] = { {1,2,3,4,5}, {2,3,4,5,6}, {3,4,5,6,7} };
	//printf("%d\n", arr[1][3]);

	 

	/*char arr1[] = "you are a beach";
	char arr2[] = "***************";

	int left = 0;
	int right = strlen(arr1)-1;

	while (left <= right)
	{
		arr2[left] = arr1[left];
		arr2[right] = arr1[right];
		printf("%s\n", arr2);
		Sleep(1000);
		system("cls");
		left++;
		right--;
	}*/

	return 0;
}