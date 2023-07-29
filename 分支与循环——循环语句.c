//#define _CRT_SECURE_NO_WARNINGS
////循环语句
//// while   for   do while
//// 我们已经掌握了if 语句
////  if(条件)
////     语句;
//// 当满足条件的情况下，if语句后的语句执行，否则不执行。但是这个语句只会执行一次
//// 但是我们发现生活中很多实际例子是：同一件事情需要完成很多次 这时候就需要我们的while循环
//// 
//// 
////   while 语法结构
////    while(表达式)
////       循环语句:
//#include <stdio.h>
//int main_03()
//{
	//int i = 1;
	//while (i<=10)
	//{
	//	if (i == 5)
	//		continue;
	//		//break;
	//	printf("%d", i);
	//	i++;
	//}




	//int ch = 0;
	//ctrl + z
	//EOF - end of file -> -1
	//while (ch = getchar() != EOF)
	//
		//putchar(ch);
	//}



	//for循环
	//for(表达式1;表达式2;表达式3)
	//   循环语句：
	//    表达式1为初始化部位，用于初始化循环变量的。 
	//    表达式2为条件判断部分，用来判断循环终止
	//    表达式3为调整部分，用于循环条件的调整 

	//int i = 0;
	//for (i = 1; i <= 10; i++)
	//{
	//	if (i == 5)
	//		continue;
	//	printf("%d", i);
	//	//初始化 判断 调整
	//}
	// 
	// 
	//for语句的循环控制变量
	//1. 不可在for循环体内修改循环变量，防止for循环失去控制
	//2. 建议for语句的循环控制变量的取值采用‘前必后开区间’的写法
	// 
	//for循环的初始化 调整 判断都可以省略
	//但是
	//for循环的额判断部分 如果被省略，那判断条件就是恒为正
	//尽量不要省略

	int x, y;
	for (x = 0, y = 0; x < 2 && y < 5; ++x, y++)
	{
		printf("hehe\n");
	}


	//do while 循环
	// do语句语法
	// 
	// do
	//   循环语句:
	//  while(表达式)；
	//
	int i = 1;
	do
	{
		if (i == 5)
			continue;
		printf("%d", i);
		i++;
	} 
	while (i <= 10);
	return 0;
}