#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <string.h>
#include <Windows.h>
//int main()
//{
//	int num1[5] = { 1,2,3,4,5 };//��ȫ��ʼ��
//	int num2[5] = { 1,2,3 };  //����ȫ��ʼ����ʣ���Ԫ�س�ʼ����0
//	int num3[] = { 1,2,3 }; //��������ʼ���ˣ��ǿ���ʡ�Ե������С��
//	//����Ĵ�С���Ǹ��ݱ�������ʼ����������ȷ����
//
//	return 0;
//}

//int main()
//{
//	//   int num1[5] = { 1,2,3,4,5,6 };//err
//	int num1[5] = { 1,2,3,4,5 }; //  num1��������   int [5]
//	int num2[10] = { 1,2,3 };  //num2�������� int [5]
//
//
//}

//int main()
//{
//	int num1[10] = { 1,2,3,4,5,6,7,8,9,10 };
//	//printf("%d\n", num1[6]);  //[]�±����ò�����
//	//int n = 10;
//	//int arr2[n];
//
//	//C99  ��׼֮ǰ����Ĵ�Сֻ���ǳ���ָ��������ʹ�ñ���
//	//C99  ֮��Ϊʲô�Ϳ���ʹ�ñ�����  -- �䳤����
//	/*for (int i = 0; i < 10; i++) {
//		printf("%d ", num1[i]);
//	}*/
//
//	for (int i = 0; i < 10; i++)
//	{
//		scanf("%d", &num1[i]);  //num1����Ҫ&
//	}
//
//	for (int i = 5; i < 10; i++)
//	{
//		printf("%d ", num1[i]);
//	}
//	return 0;
//}


//int main()
//{
//	int num1[10] = { 1,2,3,4,5,6,7,8,9,10 };
//	for (int i = 0; i < 10; i++)
//	{
//		printf("&arr[%d] = %p\n", i, &num1[i]);
//	}
//	return 0;
//}

//int main()
//{
//	int arr[10] = { 1,2,3 }; //  char  1   short 2  int 4    4* 10 = 40
//	printf("%zd\n", sizeof(arr)); //size(������)�������������ռ�ڴ�ռ�Ĵ�С����λ���ֽ�
//	printf("%zd\n", sizeof(arr[0])); 
//	printf("%zd\n", sizeof(arr) / sizeof(arr[0])); //10
//	size_t sz = sizeof(arr) / sizeof(arr[0]);
//	for (int i = 0; i < sz; i++)
//	{
//		printf("%d ",arr[i]);
//	}
//	return 0;
//}

//int main()
//{
//	int arr1[][90] = { 1,2 };
//
//	int arr2[3][5] = { 0 };
//	int arr3[3][5] = { 1,2,3,4,5,6,7 };
//
//	int arr4[3][5] = { {1,2},{3,4,5},{6,7} };
//
//	int arr[3][5] = { 1,2,3,4,5,
//		              2,3,4,5,6,
//		              3,4,5,6,7 };
//	//printf("%d\n", arr[1][3]);  //�±����ò�����
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 5; j++)
//		{
//			//scanf("%d", &arr[i][j]);
//			printf("&arr[%d][%d] = %p\n", i, j, &arr[i][j]);
//		}
//		//printf("\n");
//	}
//	return 0;
//}

//int main()
//{
//	int arr1[10];
//	int arr2[3 + 5];
//	//�䳤����
//	int n = 10;
//	//int arr[n];
//
//	//VS��Ȼ֧����C99���﷨�����ǲ���ȫ��֧�ֵ�
//	//����䳤������VS�Ͼ��ǲ�֧�ֵ�  gcc������
//	return 0;
//}


int main()
{
	//  77788888888777
	//  77777777777777
	char arr1[] = "welcome to dongbei!!!!";
	char arr2[] = "**********************";

	int left = 0;
	int right = strlen(arr1) - 1;

	while (left <= right)
	{
		arr2[left] = arr1[left];
		arr2[right] = arr1[right];
		printf("%s\n", arr2);
		Sleep(1000); //���� ����
		system("cls");
		left++;
		right--;
	}
	
	return 0;
	
}