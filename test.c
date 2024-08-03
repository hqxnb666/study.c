#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
//#include "add.h"


//先声明，后使用

//函数的定义
//定义也是一种特殊的声明
//
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
//int main()
//{
//	/*int y = 0;
//	scanf("%d", &y);
//	if (is_leap_year(y)) {
//		printf("%d 是闰年\n", y);
//	}
//	else
//	{
//		printf("%d 不是闰年\n", y);
//	}*/
//
//	int a = Add(3, 5);
//	printf("%d\n", a);
//	return 0;
//}




//int main()
//{
//	{
//		int a = 10;
//		printf("%d\n", a);
//	}
//	printf("%d\n", a);
//	return 0;
//}
// 

//{}里面的变量叫做局部变量
//{}外面的变量叫做全局变量

//int g_val = 2024;   //全局变量
//我们说全局变量可以任意使用
extern int g_val;
void test()
{
	printf("2: %d\n", g_val);
}
int main()
{
	printf("1: %d\n", g_val);
	test();
	
	return 0;
}