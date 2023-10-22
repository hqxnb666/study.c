#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
//char* my_strcat(char* dest,const char* src)
//{
//	char* ret = dest;
//	assert(dest && src);
//	while(*dest != '\0')
//	{
//		dest++;
//	}
//	while (*dest++ = *src++)
//	{
//		;
//	}
//	return ret;
//}
//
//int main()
//{
//    //strcat\
//   //进行字符串追加   abc  def     变成abcdef       
//	char arr1[20] = "abc";
//	char arr2[] = "def";
//	my_strcat(arr1, arr2);
//	puts(arr1);
//	//1.找到arr1末尾 \0
//	//2.把arr2追加到arr1后边  
//
//	//目标空间必须足够大
//	//目标空间中必须有\0
//	//源字符串也得有\0  在拷贝是将源字符串的\0拷贝过去
//	
//
//	return 0;
//}

//int my_strcmp(const char* str1,const char* str2)
//{
//	assert(str1 && str2);
//	while (*str1 == *str2)
//	{
//		if (*str1 == '\0')
//		{
//			return 0;
//		}
//		str1++;
//		str2++;
//	}
//	if (*str1 > *str2)
//		return 1;
//	else
//		return -1;
//
//}
// //strcmp -- 不是比较长度 比较对应位置上的字符大小  ASCII
//int main()
//{
//	char arr1[] = "abcdef";
//	char arr2[] = "abq";
//	int ret = my_strcmp(arr1, arr2);
//	printf("%d", ret);
//	return 0;
//}

//strcat  strcmp strcpy   长度不受限函数
//strncar  strncmp dtrncpy

//int main()
//{
//	char arr1[20] = { 0 };
//	char arr2[] = "abcdefghi";
//	strncpy(arr1, arr2, 3);
//	puts(arr1);
//	return 0;
//}
   


//strstr  返回第一次出现的位置
// strstr1没有str2  返回NULL
//const char* my_strstr(const char* str1, const char* str2)
//{
//	const char* cp;//记录开始匹配的位置
//	const char* s1; //便利str1指向的字符串
//	const char* s2;
//
//	assert(str1 && str2);
//	if ( *str2 == '\0')
//		return str1;
//	cp = str1;
//	while (*cp)
//	{
//		s1 = cp;
//		s2 = str2;
//		while (*s1 && *s2 && *s1 == *s2)
//		{
//			s1++;
//			s2++;
//		}
//		if (*s2 == '\0')
//		{
//			return cp;
//		}
//		cp++;
//	}
//	return NULL;
//}
//int main()
//{
//	char arr1[] = "abgjhifg";
//	char arr2[] = "def";
//	const char* ret = my_strstr(arr1, arr2);
//	if (ret == NULL)
//	{
//		printf("dsdsd");
//	}
//	return 0;
//}
//

//strtok  切割字符串

//int main()
//{
//	char arr[] = "jdhsjdh@sdsd.net";
//
//	char buf[200] = { 0 };
//	strcpy(buf, arr);
//	char* p = "@.";
//	char* s = NULL;
//	for (s = strtok(arr, p); s != NULL; s = strtok(NULL, p))
//	{
//		puts(s);
//	}
//	return 0;
//}



//strerror 将错误码翻译成错误信息 然后返回错误信息字符串的起始地址
// C语言使用库函数发生错误 就会把错误码放在errno的变量中 是一个去安居变量 可以直接使用
 

//perror  直接打印错误码所对应的信息

int main()
{
	
	return 0;
}
