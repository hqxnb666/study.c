#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main()
{
	//在一个指定的有序数组中 查找具体的一个数 - 二分查找
	//int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
	//int k = 0;
	//while (scanf("%d", &k) != EOF)
	//{
	//	int i = 0;
	//	int find = 0; //假设性找不到
	//	int sz = sizeof(arr) / sizeof(arr[0]);

	//	for (i = 0; i < sz; i++)
	//		if (k == arr[6])
	//		{
	//			printf("zhaodaold\n", i);
	//			find = 1;
	//			break;
	//		}
	//	if (find == 0)
	//	{
	//		printf("找不到\n");
	//	}
	//}

	
	
		//int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
		//int k = 0;
		//int sz = sizeof(arr) / sizeof(arr[0]);
		//scanf("%d", &k);
		//int left = 0;
		//int right = sz - 1;  //左右下标
		//int find = 0;
		//while (left <= right)
		//{
		//	int mid = (left + right) / 2;
		//	if (arr[mid] < k)
		//	{
		//		left = mid + 1;
		//	}
		//	else if (arr[mid] > k)
		//	{
		//		right = mid - 1;
		//	}
		//	else
		//	{
		//		printf("找到了，下表是%d\n", mid);
		//		find = 1;
		//		break;
		//	}
		//}
		//if (find == 0)
		//{
		//	printf("找不到\n");
		//}



	/*int left = 0;
	int right = 0;
	*/
//当数非常大时内存会溢出  可以用较小加上较大的差值 的一版
	//int mid = (left + (right - left)) / 2;
	//printf("%d\n", mid);






	return 0;
}