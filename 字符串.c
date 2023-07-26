#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
int main_03()
{
	//"hello";
	//"";//空字符串
	//char arrl[] = "abc";  //数组
	//"abc" --- 'a', b', 'c'、\0 ,      \0' -- 字符串的结束标志 
	//char arr2[] = { 'a', 'b', 'c' , 0};
	//printf("%s\n", arrl);
	//printf("%s\n", arr2);




	//char arr1[] = "abc";
	//char arr2[] = { 'a', 'b', 'c' , '\0'};
	//printf("%d\n", strlen(arr1)); //strlen --string length-- 计算字符串长度的
	//printf("%d\n", strlen(arr2));




	//printf("c:\\test\32\\test.c");//用\转译\让他变成一个普通的\
	//\t---水平制表符  、\n换行








	//printf("%c\n", '\'');//把'转译
	//printf("%s\n", "a");
	//printf("%d\n", strlen("c:\test\32\test.c"));
	//32--- 32是2个8进制数字
	//32作为8进制代表的那个十进制数字，作为ASCII对应的字符
	//32--》10进制 为26
	//  /ddd 1-3个只能是八进制 /xdd表示2个16进制数字





	//int input = 0;
	//printf("加入比特\n");
	//printf("你要好好学习吗?(1/0)");
	//scanf("%d", &input);//1/0
	//if (input == 1)
	//printf("给你一个好offer\n");
	//else
	//	printf("卖红薯\n");








	return 0;
} 
//   "hello bit.\n"  注： 字符串的结束标志是一个\0的转义字符  
//在计算字符串长度的时候\0是结束标志，不算做字符串内容。
//数据在计算机存储的是2进制
//#av$
//a - 97
//A - 65
//......   这种编码方式叫ASCII 编码
// ASCII 码值