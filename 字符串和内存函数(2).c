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
//   //�����ַ���׷��   abc  def     ���abcdef       
//	char arr1[20] = "abc";
//	char arr2[] = "def";
//	my_strcat(arr1, arr2);
//	puts(arr1);
//	//1.�ҵ�arr1ĩβ \0
//	//2.��arr2׷�ӵ�arr1���  
//
//	//Ŀ��ռ�����㹻��
//	//Ŀ��ռ��б�����\0
//	//Դ�ַ���Ҳ����\0  �ڿ����ǽ�Դ�ַ�����\0������ȥ
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
// //strcmp -- ���ǱȽϳ��� �Ƚ϶�Ӧλ���ϵ��ַ���С  ASCII
//int main()
//{
//	char arr1[] = "abcdef";
//	char arr2[] = "abq";
//	int ret = my_strcmp(arr1, arr2);
//	printf("%d", ret);
//	return 0;
//}

//strcat  strcmp strcpy   ���Ȳ����޺���
//strncar  strncmp dtrncpy

//int main()
//{
//	char arr1[20] = { 0 };
//	char arr2[] = "abcdefghi";
//	strncpy(arr1, arr2, 3);
//	puts(arr1);
//	return 0;
//}
   


//strstr  ���ص�һ�γ��ֵ�λ��
// strstr1û��str2  ����NULL
//const char* my_strstr(const char* str1, const char* str2)
//{
//	const char* cp;//��¼��ʼƥ���λ��
//	const char* s1; //����str1ָ����ַ���
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

//strtok  �и��ַ���

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



//strerror �������뷭��ɴ�����Ϣ Ȼ�󷵻ش�����Ϣ�ַ�������ʼ��ַ
// C����ʹ�ÿ⺯���������� �ͻ�Ѵ��������errno�ı����� ��һ��ȥ���ӱ��� ����ֱ��ʹ��
 

//perror  ֱ�Ӵ�ӡ����������Ӧ����Ϣ

int main()
{
	
	return 0;
}
