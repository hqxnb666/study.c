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
//	//strlen   �⺯�� �������ַ�������
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
//		printf("<\n");//�����޷�������������޷���   һ��>0s
//	return 0;
//}



int main()
{
	//ģ��ʵ��
	//1.������
	//2.�ݹ�
	//3��ָ��-ָ��

	//strcpy
	char arr1[20] = { 0 };
	char arr2[] = "abc";
	my_strcpy(arr1, arr2);
	puts(arr1);
	return 0;
}