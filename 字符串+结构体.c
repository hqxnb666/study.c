#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>


//int main()
//{
//	//�ַ�����
//	//1.�ַ����ຯ��
//	//2.�ַ�ת������
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


//�ַ�ת��

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
//�ڴ���غ���
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
//		//��ǰ���
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
//		//�Ӻ���ǰ
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



//memset ���ֽ�Ϊ��λ
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
}s1,s2,s3;//  �ṹ�����  //ȥ���ӱ���
struct Book
{
	char name[20];
	char author[20];
	float price;
}b1;
struct   //�����ṹ�� 
{
	char name[20];
	char author[20];
	float price;
}b1;
int main()
{
	struct student s4, s5, s6;//�ֲ�����
	return 0;
}