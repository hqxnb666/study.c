#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include<assert.h>
//void my_strcpy(char* dest, char* src)
//{
//	while (*src!='\0')
//	{
//		*dest = *src;
//		dest++;
//		src++;
//	}
//	*dest = *src;
//}
//
//void my_strcpy(char* dest, char* src)
//{
//	while (*dest++ = *src++)
//	{
//		;
//		
//	}
//	*dest = *src;
//}
size_t my_strlen(const char* arr)
{
	int i = 0;
	assert(arr != 0);
	while (*arr != '\0')
	{
		arr++;
		i++;
	}
	return i;
}


//void my_strcpy(char* dest, char* src)
//{
//	assert(dest != NULL);//断言
//	assert(src != NULL);
//	while (*dest++ = *src++)
//	{
//		;
//
//	}
//	*dest = *src;
//}

int main()
{
	//优秀代码：
	//1.运行正常
	//2.bug很少
	//3.效率高
	//4.可读性高
	//5.可维护性高
	//6.注释清晰

	//常见coding技巧
	//使用assert  尽量使用const   养成良好风格

	/*char arr1[20] = "xxxxxxxxxxxxx";
	char arr2[] = "heelo bit";
	 my_strcpy(arr1, arr2);
	printf("%s\n", arr1);*/

	/*const int num = 20;
	num = 30;*/
	//const修饰变量 是在语法层面限制const修改
	//本质上num还是不按量 是一种不能被修改的变量
	/*const int num = 10;
	printf("num = %d\n", num);

	int* p = &num;
	*p = 20;
	printf("num=%d", num);*///这就能非法访问了
	//const 放在*的左边
	//const 放在*的右边 含义是不一样的

	//模拟实现一个strlen函数
	char arr[] = "abcdef";
	size_t len = my_strlen(arr);
	printf("%d", len);

	return 0;
}