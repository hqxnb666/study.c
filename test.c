#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
//int main()
//{
//	int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
//	//            0 1 2 3 4 5 6 7 8 9
//	int k = 0;
//	scanf("%d", &k);
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	int find = 0;
//	for (int i = 0; i < sz; i++)
//	{
//		if (k == arr[i])
//		{
//			printf("找到了，下标是%d\n", i);
//			find = 1;
//			break;
//		}
//	}
//	if (find == 0)
//	{
//		printf("找不到\n");
//	}
//	return 0;
//}

//int main()
//{
//	int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
//	int k = 0;
//	scanf("%d", &k);
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	//查找--二分查找
//	int find = 0;
//	int left = 0;
//	int right = sz - 1;
//	while (left <= right)
//	{
//
//		int mid = left + (right-left) / 2;
//		
//		//_CRT_INT_MAX;
//		if (arr[mid] < k)
//		{
//			left = mid + 1;
//		}
//		else if (arr[mid] > k)
//		{
//			right = mid - 1;
//		}
//		else
//		{
//			printf("找到了，下标是:%d\n", mid);
//			find = 1;
//			break;
//		}
//	}
//	if (find == 0)
//	{
//		printf("找不到了\n");
//	}
//	
//	return 0;
//}


//int main()
//{
//	int sum = 0;
//	int num[10] = {0};
//	for (int i = 0; i < 10; i++)
//	{
//		scanf("%d", &num[i]);
//		sum += num[i];
//	}
//	double average = sum / 10.0;
//	printf("平均值为：%lf", average);
//	return 0;
//}

//int main()
//{
//	int arr1[10] = { 0 };
//	int arr2[10] = { 0 };
//	for (int i = 0; i < 10; i++)
//	{
//		scanf("%d", &arr1[i]);
//		
//
//	}
//	for (int i = 0; i < 10; i++)
//	{
//		scanf("%d", &arr2[i]);
//
//	}
//	for(int i = 0; i<10; i++)
//	{
//		int temp = arr1[i];
//		arr1[i] = arr2[i];
//		arr2[i] = temp;
//	}
//
//	return 0;
//
//}


//int main()
//{
//	int a = 99;
//	int b = 11;
//	int c = 0;
//
//	while (c=a%b)
//	{
//		a = b;
//		b = c;
//	}
//	printf("%d\n", b);
//	return 0;
//}

int main()
{
	for (int year = 1000; year <= 2000;year++)
	{
		if (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0))
		{
			printf("%d ", year);
		}
	}

	
	return 0;
}
