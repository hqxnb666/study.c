#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <assert.h>
char* my_strcpy(char* dest, const char* src)

{
	char* ret = dest;
	assert(dest && src);
	while (*dest++ = *src++)
	{
		
	}
	return ret;
}
//int main()
//{
//	//strlen   库函数 用来求字符串长度
//	const char* str = "abcdef";
//	size_t len = strlen("abcdef");
//	size_t len2 = strlen(str);
//	printf("%d\n", len);
//	printf("%d\n", len2);
//	return 0;
//}


//int main()
//{
//	if (strlen("abc") - strlen("abcdef") > 0)
//		printf(">=\n");
//	else
//		printf("<\n");//两个无符号数相减还是无符号   一定>0s
//	return 0;
//}



int main()
{
	//模拟实现
	//1.计数器
	//2.递归
	//3。指针-指针

	//strcpy
	char arr1[20] = { 0 };
	char arr2[] = "abc";
	my_strcpy(arr1, arr2);
	puts(arr1);
	return 0;
}