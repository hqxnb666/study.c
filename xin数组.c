#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int mn()
{
	//������Ԫ�ز���Ϊ0 ��Ŷ������ ������ͬ

	//�����﷨����
	//type arr_name[����ֵ]
	// type ������  char short int ....
	//arr_name ���������
	//[]��ʾ�����С


	//int num[5] = { 1,2,3,4,5 };//��ȫ��ʼ��
	//int num2[5] = { 1,2,3 };//����ȫ ʣ��Ϊ0
	//int num3 [] = {1,2,3};//��������ʼ���� �ǿ���ʡ�������
	
	//�����������͵� arr1 ��������  int [10] ch����������� char [5]



	//һά����

	//�±����ò�����
	//arr[6]


	/*int num1[10] = { 1,2,3,4,5,6,7,8,9,10 };
	int i = 0;
	for (i = 0; i < 10; i++)
		printf("&arr[%d] = %p\n", &num1[i]);*/ // %p��ӡ��ַ
	 //һά�����ڴ�����������ŵ� �����±������ ��ַ�ɵ͵��߱仯


	//int arr[10] = { 1,2,3 };
	//printf("%d\n", sizeof(arr));//  sizeof ������������С ��λ���ֽ�
	//printf("%d\n", sizeof(arr[0]));
	//printf("%d\n", sizeof(arr) / sizeof(arr[0]));

	//int sz = sizeof(arr) / sizeof(arr[0]);




		
	

	// ��ά����
	// tyoe arr_name[����ֵ1][����ֵ2]
	// 
	// int arr[3][5]
	// double data[2][8]
	// 3��ʾ���������� 5��ʾÿһ�����Ԫ��

	//int arr[3][5] = { {1,2,3,4,5}, {2,3,4,5,6}, {3,4,5,6,7} };
	//printf("%d\n", arr[1][3]);

	 

	/*char arr1[] = "you are a beach";
	char arr2[] = "***************";

	int left = 0;
	int right = strlen(arr1)-1;

	while (left <= right)
	{
		arr2[left] = arr1[left];
		arr2[right] = arr1[right];
		printf("%s\n", arr2);
		Sleep(1000);
		system("cls");
		left++;
		right--;
	}*/

	return 0;
}