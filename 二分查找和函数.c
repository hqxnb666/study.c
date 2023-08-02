#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int is_leap_year(int y)
{
	if ((y % 4 == 0 && y % 100 != 0) || (y % 400 == 0))
		return 1;
	else
		return 0;
}

int get_days_ofmonth(int y, int m)
{
	int days[13] = { 0,31,28,31,30,31,30,31,31,31,31,30,31 };
 //                    1  2  3  4  5  6  7  8  9  
int d = days[m];
 if (is_leap_year(y) && m == 2)
{
	d += 1;
	return d;
}
//void set_arr(int arr[], int sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		arr[i] = -1;
//	}
//}
//void print_arr(int arr[], int sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%d", arr[i]);
//	}
//	printf("\n");
//}
//int Add(int x, int y)
//{
//	int z = 0;
//	z = x + y;
//	return z;
//}

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



	//自定义函数

	//ret_type fun_name(形式参数)
	//{
	// 函数返回类型两个  1：void 2：其他
	//}

	
	
	//int a = 0;
	//int b = 0;
	//scanf("%d %d", &a, &b);
	////计算求和
	//int c = Add(a, b);
	////shuchu
	//printf("%d\n", c);


	//函数的参数分为 实参和形参

	//实参



	//return 语句
	//return x+y //先算出结果 再返回
	//return 返回一次后面的语句不再执行
	//if分支必须有完整的return



	/*int arr[10] = { 1,2,3,4,5,6,7,8,9,10 };
	int sz = sizeof(arr) / sizeof(arr[0]);
	set_arr(arr, sz);
	print_arr(arr, sz);*/


	//zhongdain
	// 1 形参要与实参个数匹配
	// 2 函数的实参是数组 形参也可以写成数组形式
	// 3 形参如果是一堆数组 数组大小可以不写
	// 4 形参如果是二维数组 行可以省略 列不能
	// 5 形参操作的数组可能与实参是同一个数组


	//嵌套调用和链式访问
	int y = 0;//年
	int m = 0;//月
	scnaf("%d %d", &y, &m);
	int d = get_days_of_month(y, m);
	return 0;
}