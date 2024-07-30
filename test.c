#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <math.h>
//int main()
//{
//	double r = sqrt(-16.0);
//	printf("%lf\n", r);
//	return 0;
//	 int a;
//}


//函数的返回类型有两类：
//1.  void - 表示说明都不返回
//2.  其他类型  int char short
//void menu()
//{
//	printf("************************\n");
//	printf("************************\n");
//	printf("************************\n");
//	printf("************************\n");
//	printf("************************\n");
//
//}
//int main()
//{
//	menu();
//}
//int Add(int x, int y) //形式上存在，简称形参
//{
//	
//	return (x+y);
//}
//
//int main()
//{
//	int a = 0;
//	int b = 0;
//	scanf("%d%d", &a, &b);
//	//计算求和
//	int c = Add(a, b);//a 和 b 是真实传递给Add的参数，是实际参数，简称实参
//	//输出
//	printf("%d\n", c);
//}

//void test()
//{
//	int n = 0;
//	scanf("%d", &n);
//	printf("hehe\n");
//	if (n == 5)
//		return;
//	printf("haha\n");
//}


//int test1()
//{
//	return 3.14;
//}

//int test2()
//{
//	int n = 0;    //正常
//	if (n == 5)
//	{
//		return 1;
//	}
//		//如果没有if，会怎么办
//}
//int main()
//{
//	int n = test2();
//	printf("%d\n", n);
//	return 0;
//}


//void set_arr(int arr[3][5], int r, int l)
//{
//	/*int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		arr[i] = -1;
//	}*/
//}
//void print_arr(int arr[],int  sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%d ", arr[i]);
//	}
//	printf("\n");
//}
//int main()
//{
//	int arr1[10] = { 1,2,3,4,5,6,7,8,9,10 };
//	int arr2[3][5] = { 1,2,3,4,5,2,3,4,5,6,3,4,5,6,7 };
//	int r = 3;
//	int l = 5;
//	int sz = sizeof(arr2) / sizeof(arr2[0]);
//	//先写一个函数，把arr中的内容全部设置为-1;
//	set_arr(arr2,r,l);
//	//写一个函数，把arr中的内容打印出来
//	
//	print_arr2(arr1,sz);
//	return 0;
//}

//int is_leap_year(int y)
//{
//	if ((y % 4 == 0) && (y % 100 != 0) || (y % 400 == 0))
//		return 1;
//	else
//		return 0;
//}
//int get_days_of_month(int y, int m)
//{
//
//	int days[13] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
//	//               0 1   2  3  4 
//	int d = days[m];
//	
//	if (is_leap_year(y) && m == 2)
//	{
//		d += 1;
//	}
//	return d;
//}
////函数是可以嵌套调用的，但是不能嵌套定义
//int main()
//{
//	int y = 0; //年
//	int m = 0; //月
//	scanf("%d %d", &y, &m);
//	int d = get_days_of_month(y, m);
//	printf("%d\n", d);
//	return 0;
//}
//#include <string.h>
//int main()
//{
//	printf("%zd\n", strlen("abc"));
//	return 0;
//}



//是闰年，返回1
//不是闰年，返回0
 

int is_leap_year(int y)
{
	if ((y % 4 == 0) && (y % 100 != 0) || (y % 400 == 0))
		return 1;
	else
		return 0;
}

//函数的定义也是一种特殊的声明
int main()
{
	int y = 0;
	scanf("%d", &y);
	if(is_leap_year(y))
	{
		printf("%d 是闰年\n", y);
	}
	else
	{
		printf("%d不是闰年\n", y);
	}

	return 0;
}
//函数的定义
