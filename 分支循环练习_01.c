//#define _CRT_SECURE_NO_WARNINGS//�ж����Ƿ�Ϊ����   ���1-100������
//#include <stdio.h>
//#include <Windows.h>
//#include <stdlib.h>
//int main_02()
//{
//	int i = 0;
//	while (i <= 100)
//	{
//		if (i % 2 == 1)
//			printf("%d", i);
//		i++;
//	}//��һ��д��
//
//
//
//	int i = 1;
//	while (i <= 100)
//	{
//		printf("%d", i);
//		i + 2;
//	}
//
//
//	//����n�Ľ׳�
//
//
//	
//	int i = 0;
//	int n = 0;
//	int ret = 1;
//	scanf("%d", &n);
//	for (i = 1; i <= n; i++)
//	{
//		ret = ret * i;
//
//	}
//	printf("%d\n", ret);
//
//
//	//����1!+ 2! + 3! +...
//	int i = 0;
//	int n = 0;
//	int ret = 1;
//	int sum = 0;
//	
//	for (n = 1; n <= 3; n++)
//	{
//	
//			ret = ret * n;
//		
//	//n�Ľ׳�
//		sum = sum + ret;
//	}
//    printf("sum = %d\n", sum);
//
//
//
//
//	// 3.��һ�����������в��Ҿ����ĳ������n  ��дint binsearch(int x, int v[], int n);���ܣ���v[0]<=v[1]<=[2]....<=
//	//v[n-1]�������в���x
//
//	int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
//	int k = 7;
//	int i = 0;
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	for (i = 0; i < sz; i++)
//	{
//		if (k == arr[i])
//		{
//			printf("�ҵ��ˣ��±��ǣ�%d\n", i);
//			break;
//		}
//
//	}
//	if (i = sz)
//		printf("�Ҳ���\n");
//
//
//
//	//4.��д���� ��ʾ����ַ��������ƶ������м���
//	
//	char arr1[] = "welcome to bit!!!!!";
//	char arr2[] = "###################";
//	int left = 0;
//	int right = strlen(arr1)-1;
//	while (left<=right)
//	{
//		arr2[left] = arr1[left];
//		arr2[right] = arr1[right];
//		printf("%s\n", arr2);
//		//��Ϣһ�� ����ͷ�ļ�
//		Sleep(1000);
//		system("cls");//cls �����Ļ ����ͷ�ļ�
//		left++;
//		right--;
//	}
//	printf("%s\n", arr2);
//
//
//
//
//	//5.��д����ʵ�֣�ģ���û���½�龰������ֻ�ܵ�½���� ��ֻ���������������룬�����ȷ����ʾ�ɹ���������ξ��������˳�����
//
//	int i = 0;
//	char password[20] = { 0 };
//	for (i = 0; i < 3; i++)
//	{
//		printf("����������:");
//		scanf("%s", password);
//		if (strcmp(password, "123456") == 0)// ==���������Ƚ������ַ������  Ӧ����һ���⺯�� strcmp
//		{
//			printf("��¼�ɹ�\n");
//			break;
//		}
//		else
//			printf("�������\n");
//	}
//	if (i == 3)
//		printf("��������������˳�����\n");
//	return 0;
//}