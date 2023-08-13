#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int _04()
{
	/*int a = 0;
	int b = 0;
	int c = 0;
	scanf("%d %d %d", &a, &b, &c);
	if (a < b)
	{
		int tmp = a;
		a = b;
		b = tmp;
	}
	if (a < c)
	{
		int tmp = a;
		a = c;
		c = tmp;
	}
	if (b < c)
	{
		int tmp = b;
		b = c;
		c = tmp;
	}
	printf("%d %d %d\n", a, b, c);*///三个数从大到小输出





	//打印3的倍数的数
	/*int i = 0;
	for (i = 1; i <= 100; i++)
	{
		if (i % 3 == 0)
			printf("%d\n", i);
	}*/




	//最大公约数  给定两个数 求这两个数的最大公约数
	//辗转相除法
	/*int m = 24;
	int n = 18;
	int r = 0;
	while(m%n)
	{
		r = m % n;
		m = n;
		n = r;
	}
	printf("%d\n", n);*/

	//int count = 0;

	////打印1000·2000的闰年
	//int year = 0;
	//for (year = 1000; year <= 2000; year++)
	//{
	//	//判断闰年的规则 能被4整除 不能被100整除  2. 能被400整除是闰年
	//	int count = 0;
	//	if (year % 4 == 0 && year % 100 != 0)
	//	{
	//		printf("%d", year);
	//		count++;
	//	}
	//	else if (year % 400 == 0)
	//	{
	//		printf("%d", year);
	//		count++;
	//	}
	//}
	//printf("\ncount = %d\n", count);




	//打印100-200的素数
	//int i = 0;
	//int count = 0;
	//for (i = 100; i <= 200; i++)
	//{
	//	//判断i是否为素数
	//	//素数判断规则 1.试除法 产生2- i-1的数字
	//	int j = 0;
	//	for (j = 2; j < i; j++)
	//	{
	//		if (i % j == 0)
	//		{
	//			break;
	//		}
	//	}
	//	if (j == i)
	//	{
	//		printf("%d", i);
	//		count++;
	//	}
	//}
	//printf("\ncount = %d\n", count);//如果不是素数 那可以写成 i = a*b  a/b中至少有一个数字是<=开平方i 16=2*8 =4*4
	          //int j = 0;
	//         for(j=2; j<=sqrt(); j++)             sqrt- 开平方的数学库函数 要加头函数 #include <math.h>
	//                  if(j>sqrt（i）;





   //数9的个数  编写一下1-100的整数出现多少个9
//int i = 0;
//int count = 0;
//for (i = 1; i <= 100; i++)
//{
//	if (i % 10 == 9)
//		count++;
//    if (i / 10 == 9)
//		count++;
//}
//printf("%d\n", count);
   




 //计算1/1-1/2+1/3-1/4+1/.......+1/99-1/100
//int i = 0;
//double sum = 0.0;
//int flag = 1;
//for (i = 1; i <= 100; i++)
//{
//	sum += flag * 1.0 / i;
//	flag = -flag;
//}
//printf("%lf\n", sum);



     //求十个数的最大值


//int arr[] = { -1,-2,-3,-4,-5,-6,-7,-8,-9,-10 };
//int max = arr[0];
//int i = 0;
//int sz = sizeof(arr) / sizeof(arr[0]);
//for (i = 1; i < sz; i++)
//{
//	if (arr[i] > max)
//	{
//		max = arr[i];
//	}
//}
//printf("max = %d\n", max);



   //输出9*9的乘法口诀
//int i = 0;
//for (i = 1; i <= 9; i++)
//{
//	//打印一行
//	int j = 1;
//	for (j = 1; j <= i; j++)
//	{
//		printf("%d*%d=%2d", i, j, i * j); //%2d -2d是向左对齐和向右对齐
//	}
//	printf("\n");
//}


  //goto语句  从理论上讲goto是没有必要的 但是某些场合还是用的着的 最常见的用法就是终止程序在某些深度嵌套的结构的处理过程
// 例如一次跳出基层循环或多种循环 这种情况break是达不到目的的
  
   
    //shutdown -s -t 60 关机
     //system 执行系统命令
char input[20] = { 0 };
system("shutdown -s -t 60");
again:
printf("前请注意你的电脑将在一分内关机，如果输入：我是猪，就取消关机\n请输入：");
scanf("%s", &input);
if (strcmp(input, "我是猪") == 0)//比较两字符串-strcmp()
{
	system("shutdown -a");
}
else
{
	goto again;
}
   
	return 0;
}