#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//int main()
//{
//	int a = 10;
//	int arr[] = { 1,2,3,4,5,6,7 };
//	//c���Ը�����Աһ��Ȩ�����ܹ���̬��������ڴ�ռ�
//
//	//malloc �� free
//	//����һ��ռ� ���10������
//	//int* p = (int*)malloc(10 * sizeof(int));
//	//������ؿ�ָ�� ����ʧ��
//	int* p = (int*)malloc(INT_MAX*4);
//	if (p == NULL)
//	{
//		perror("malloc");
//		return 1;
//	}
//	 //ʹ��
//	int i = 0;
//	for (i = 0; i < 10; i++)
//	{
//		*(p + i) = i;
//	}
//	free(p);
//	p = NULL;
//
//	//malloc����Ŀռ���ô�ͷ���
//	// 1.free�ͷ� -- ����
//	// �����˳���  malloc����Ŀռ�Ҳ�ᱻ����ϵͳ���յ� == ����
//
//	//���������  ˭����Ŀռ� ˭ȥ�ͷ�  �Լ����ͷ� ҲҪ���������ͷ�
//	return 0;
//}




//callo
// int main()
//{
	//callocҲ������ռ��
	//calloc(10, sizeof(int)
	//  malloc(10 * sizeof(int));

	//calloc  ���˲��������� �����֮�� �Ὣ�ռ��ʼ��  ��malloc����
//	int* p = calloc(10 , sizeof(int));
//	if (p == NULL)
//	{
//		perror("malloc");
//		return 1;
//	}
//	int i = 0;
//	for (i = 0; i < 10; i++)
//	{
//		printf("%d\n", *(p + i));
//	}
//	free(p);
//	p = NULL;
//
//	return 0;
//}



//realloc    �ö�̬�ڴ�������
// ���realloc���ٿռ�ʧ��  Ҳ�᷵��NULL  �������ǲ���ֱ��ʹ��ԭ����ָ�����realloc
//int* ptr = (int*)realloc(p, 20 * sizeof(int));
//if (ptr != NULL)
//{
//	p = ptr;
//}
//free(p);
//p =NULL

//int main()
//{
//	int* p = (int*)realloc(NULL, 40); //  == malloc(40);
//	return 0;
//}



