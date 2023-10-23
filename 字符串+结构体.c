#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>


//int main()
//{
//	//字符函数
//	//1.字符分类函数
//	//2.字符转化函数
//	int ret = islower('X');
//	printf("%d", ret);//  islower -- ctype.h
//
//	return 0;
//}



//int main()
//{
//	//  isdigit   isxdigit
//	return 0;
//}


//字符转换

//tolower  toupper

//int main()
//{
//	int ret = toupper('a');
//	printf("%c\n", ret);
//	ret = tolower(ret);
//	printf("%c\n", ret);
//	return 0;
//}

//void* my_memcpy(void* str1, const void* str2, size_t sz)
//{
//	assert(str1 && str2);
//	while (sz--)
//	{
//		*(char*)str1 = *(char*)str2;
//		str1 = (char*)str1 + 1;
//		str2 = (char*)str2 + 1;
//	}
// }
//内存相关函数
//memcoy   memove  memset   memcmp
//int main()
//{
//	int arr1[10] = { 0 };
//	int arr2[] = { 'a', 'b', 'c' };
//	my_memcpy(arr1, arr2, 3);
//	return 0;
//}



//void* my_memmove(void* dest, const void* src, size_t sz)
//{
//	void* ret = dest;
//	if (dest < src)
//	{
//		//从前向后
//		int i = 0;
//		for (i < 0; i < sz; i++)
//		{
//			*(char*)dest = *(char*)src;
//			dest = (char*)dest + 1;
//			src = (char*)src + 1;
//
//		}
//
//	}
//	else
//	{
//		//从后向前
//		while (sz--)
//		{
//			*((char*)dest +sz) = *((char*)src + sz);
//		}
//	}
//	return ret;
//}
//int main()
//{
//	int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
//	my_memmove(arr, arr + 2, 20);
//	int i = 0;
//	for (i = 0; i < 10; i++)
//	{
//		printf("%d\n", arr[i]);
//	}
//	return 0;
//}



//memset 以字节为单位
//int main()
//{
//	/*char arr[] = "helloword";
//	memset(arr + 6, 'x', 3);
//	printf("%s", arr);*/
//
//
//	int arr[10] = { 0 };
//
//	return 0;
//}


//memcmp
//int main()
//{
//	int arr1[] = { 1,2,3,4,5,6,7 };
//	int arr2[] = { 1,2,3,7 };
//	int ret =memcmp(arr1, arr2, 13);
//	printf("%d\n", ret);
//	return 0;
//}










struct student
{
	char name[20];
	int age;
	char sex[5];
	float score;
}s1,s2,s3;//  结构体变量  //去安居变量
struct Book
{
	char name[20];
	char author[20];
	float price;
}b1;
struct   //匿名结构体 
{
	char name[20];
	char author[20];
	float price;
}b1;
int main()
{
	struct student s4, s5, s6;//局部变量
	return 0;
}