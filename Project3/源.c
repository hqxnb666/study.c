﻿#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main()
{
	//1.ASCII码值
	   /*  字符A - Z的ASCII码值从65 - 90

        .字符a - z的ASCII码值从97 - 122

		.对应的大小写字符的ASCII码值的差值是32

		.数字字符0 - 9的ASCII码值从48 - 57

		.换行\n的ASCII码值是：10

		.在这些字符中ASCII码值从0 - 31这以下是32个字符是不可打印字符，无法在屏幕上观*/



	/*转义字符
		\'  :用于表示字符常量'

		\\ ：用于表示一个反斜杠，防止被解释为一个转义序列符、

		\a ：警报，电脑会响

		\b：退格键，光标退回一个字符，但不删除字符

		\f：换页符

		\n：换行符

		\r：回车符，光标回到同一行的开头

		\t：制表符，光标移动到下一个水平制表位，通常是下一个 8 的倍数

		\v：垂直分割符，光标移动到下一个垂直制表位

		\ddd：d d d表示1 - 3个八进制数字

		\xdd：d d 表示2个十六进制数字*/


	/*C语言的代码是由一条一条的语句构成的，可以分成五类

		1.空语句  2.表达式语句 3.函数调用语句 4.复合语句 5.控制语句

		控制语句用于控制程序的执行流程，以实现程序的各种结构方式（C语言支持三种结构：顺序结构，选择结构，循环结构）

		可分成以下三类：

		1.条件判断语句：if语句和 switch语句

		2.循环执行语句:  do while语句 while语句，for语句

		3.转向语句： break语句， goto语句， continue语句 return语句*/



	//-Bool    使用布尔类型头文件为<stdbool.h>

	//	1 #define bool - Bool

	//	2

	//	3 #define false 0

	//	4 #define true 1





	/*++和--分为前置++和后置--

		int a = 10;

	int b = ++a;

	printf("a=%d'' b=%d, a,b);

		计算口诀：先 + 1，后使用；  先把a + 1 变成11  再赋值给b

		a++则是  先使用 后 + 1*/

	return 0;
}