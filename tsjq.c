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
//	assert(dest != NULL);//����
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
	//������룺
	//1.��������
	//2.bug����
	//3.Ч�ʸ�
	//4.�ɶ��Ը�
	//5.��ά���Ը�
	//6.ע������

	//����coding����
	//ʹ��assert  ����ʹ��const   �������÷��

	/*char arr1[20] = "xxxxxxxxxxxxx";
	char arr2[] = "heelo bit";
	 my_strcpy(arr1, arr2);
	printf("%s\n", arr1);*/

	/*const int num = 20;
	num = 30;*/
	//const���α��� �����﷨��������const�޸�
	//������num���ǲ����� ��һ�ֲ��ܱ��޸ĵı���
	/*const int num = 10;
	printf("num = %d\n", num);

	int* p = &num;
	*p = 20;
	printf("num=%d", num);*///����ܷǷ�������
	//const ����*�����
	//const ����*���ұ� �����ǲ�һ����

	//ģ��ʵ��һ��strlen����
	char arr[] = "abcdef";
	size_t len = my_strlen(arr);
	printf("%d", len);

	return 0;
}