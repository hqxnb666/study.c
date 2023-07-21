//字面常量
//const修饰的常变量
//#define定义的标识符常量
//枚举常量



#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main_02()
{
	/*int num = 4;
	3;
	100;//字面常量*/


    //const - 常属性
	//const修饰的长变量   num 不能改了
	//const int num = 4;
	//printf("%d\n", num);
	//num = 8;
	//printf("%d\n", num);





     //const	int n = 10;//n是变量，但是有常属性，所以我们说n是常变量
	//int arr[n] = { 0 };





	//#define定义的标识符常量
   // #define MAX 10

	//int arr[MAX] = { 0 };
	//printf("%d\n", MAX);//输出结果是10





	//枚举常量--一一列举
	//性别：男， 女，保密
	//三原色：红 黄 蓝
	
	//枚举关键字--enum

	enum Sex
	{
		MALE,
		FEMELE,
		SECRET,


	};
	
	printf("%d\n", MALE);//0
	printf("%d\n", FEMELE);//1
	printf("%d\n", SECRET);//2
	//枚举常量不可更改

	return 0;
}