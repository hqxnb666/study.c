#define _CRT_SECURE_NO_WARNINGS
//���飺һ����ͬ����Ԫ�صļ���
//���鶨�� 
//   int arr[10] = { 1,2,3,4,5 }; ����һ���������飬����10��Ԫ��
#include <stdio.h>
int main_06()
{
	/*int a = 1;
	int b = 2;
	int c = 3;*/
	int arr[10] = { 1,2,3,4,5,6,7,8,9,10 };//����һ�����10���������ֵ�����
	//printf("%d\n", arr[4]);  //���±�ķ�ʽ����Ԫ��
	int i = 0;
	while (i < 10)
	{
		printf("%d", arr[i]);
		i++;
	}

	/*char ch[20];
	float arr2[5];*/


	return 0;
}