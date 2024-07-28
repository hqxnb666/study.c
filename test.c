#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <string.h>
#include <Windows.h>
//int main()
//{
//	int num1[5] = { 1,2,3,4,5 };//完全初始化
//	int num2[5] = { 1,2,3 };  //不完全初始化，剩余的元素初始化成0
//	int num3[] = { 1,2,3 }; //如果数组初始化了，是可以省略掉数组大小的
//	//数组的大小，是根据编译器初始化的内容来确定的
//
//	return 0;
//}

//int main()
//{
//	//   int num1[5] = { 1,2,3,4,5,6 };//err
//	int num1[5] = { 1,2,3,4,5 }; //  num1的类型是   int [5]
//	int num2[10] = { 1,2,3 };  //num2的类型是 int [5]
//
//
//}

//int main()
//{
//	int num1[10] = { 1,2,3,4,5,6,7,8,9,10 };
//	//printf("%d\n", num1[6]);  //[]下标引用操作符
//	//int n = 10;
//	//int arr2[n];
//
//	//C99  标准之前数组的大小只能是常量指定，不能使用变量
//	//C99  之后为什么就可以使用变量了  -- 变长数组
//	/*for (int i = 0; i < 10; i++) {
//		printf("%d ", num1[i]);
//	}*/
//
//	for (int i = 0; i < 10; i++)
//	{
//		scanf("%d", &num1[i]);  //num1不需要&
//	}
//
//	for (int i = 5; i < 10; i++)
//	{
//		printf("%d ", num1[i]);
//	}
//	return 0;
//}


//int main()
//{
//	int num1[10] = { 1,2,3,4,5,6,7,8,9,10 };
//	for (int i = 0; i < 10; i++)
//	{
//		printf("&arr[%d] = %p\n", i, &num1[i]);
//	}
//	return 0;
//}

//int main()
//{
//	int arr[10] = { 1,2,3 }; //  char  1   short 2  int 4    4* 10 = 40
//	printf("%zd\n", sizeof(arr)); //size(数组名)计算的是数组所占内存空间的大小，单位是字节
//	printf("%zd\n", sizeof(arr[0])); 
//	printf("%zd\n", sizeof(arr) / sizeof(arr[0])); //10
//	size_t sz = sizeof(arr) / sizeof(arr[0]);
//	for (int i = 0; i < sz; i++)
//	{
//		printf("%d ",arr[i]);
//	}
//	return 0;
//}

//int main()
//{
//	int arr1[][90] = { 1,2 };
//
//	int arr2[3][5] = { 0 };
//	int arr3[3][5] = { 1,2,3,4,5,6,7 };
//
//	int arr4[3][5] = { {1,2},{3,4,5},{6,7} };
//
//	int arr[3][5] = { 1,2,3,4,5,
//		              2,3,4,5,6,
//		              3,4,5,6,7 };
//	//printf("%d\n", arr[1][3]);  //下标引用操作符
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 5; j++)
//		{
//			//scanf("%d", &arr[i][j]);
//			printf("&arr[%d][%d] = %p\n", i, j, &arr[i][j]);
//		}
//		//printf("\n");
//	}
//	return 0;
//}

//int main()
//{
//	int arr1[10];
//	int arr2[3 + 5];
//	//变长数组
//	int n = 10;
//	//int arr[n];
//
//	//VS虽然支持了C99的语法，但是不是全部支持的
//	//比如变长数组在VS上就是不支持的  gcc编译器
//	return 0;
//}


int main()
{
	//  77788888888777
	//  77777777777777
	char arr1[] = "welcome to dongbei!!!!";
	char arr2[] = "**********************";

	int left = 0;
	int right = strlen(arr1) - 1;

	while (left <= right)
	{
		arr2[left] = arr1[left];
		arr2[right] = arr1[right];
		printf("%s\n", arr2);
		Sleep(1000); //休眠 毫秒
		system("cls");
		left++;
		right--;
	}
	
	return 0;
	
}