#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int is_leap_year(int y)
{
	if ((y % 4 == 0 && y % 100 != 0) || (y % 400 == 0))
		return 1;
	else
		return 0;
}

int get_days_ofmonth(int y, int m)
{
	int days[13] = { 0,31,28,31,30,31,30,31,31,31,31,30,31 };
 //                    1  2  3  4  5  6  7  8  9  
int d = days[m];
 if (is_leap_year(y) && m == 2)
{
	d += 1;
	return d;
}
//void set_arr(int arr[], int sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		arr[i] = -1;
//	}
//}
//void print_arr(int arr[], int sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%d", arr[i]);
//	}
//	printf("\n");
//}
//int Add(int x, int y)
//{
//	int z = 0;
//	z = x + y;
//	return z;
//}

int main()
{
	//��һ��ָ�������������� ���Ҿ����һ���� - ���ֲ���
	//int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
	//int k = 0;
	//while (scanf("%d", &k) != EOF)
	//{
	//	int i = 0;
	//	int find = 0; //�������Ҳ���
	//	int sz = sizeof(arr) / sizeof(arr[0]);

	//	for (i = 0; i < sz; i++)
	//		if (k == arr[6])
	//		{
	//			printf("zhaodaold\n", i);
	//			find = 1;
	//			break;
	//		}
	//	if (find == 0)
	//	{
	//		printf("�Ҳ���\n");
	//	}
	//}

	
	
		//int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
		//int k = 0;
		//int sz = sizeof(arr) / sizeof(arr[0]);
		//scanf("%d", &k);
		//int left = 0;
		//int right = sz - 1;  //�����±�
		//int find = 0;
		//while (left <= right)
		//{
		//	int mid = (left + right) / 2;
		//	if (arr[mid] < k)
		//	{
		//		left = mid + 1;
		//	}
		//	else if (arr[mid] > k)
		//	{
		//		right = mid - 1;
		//	}
		//	else
		//	{
		//		printf("�ҵ��ˣ��±���%d\n", mid);
		//		find = 1;
		//		break;
		//	}
		//}
		//if (find == 0)
		//{
		//	printf("�Ҳ���\n");
		//}



	/*int left = 0;
	int right = 0;
	*/
//�����ǳ���ʱ�ڴ�����  �����ý�С���Ͻϴ�Ĳ�ֵ ��һ��
	//int mid = (left + (right - left)) / 2;
	//printf("%d\n", mid);



	//�Զ��庯��

	//ret_type fun_name(��ʽ����)
	//{
	// ����������������  1��void 2������
	//}

	
	
	//int a = 0;
	//int b = 0;
	//scanf("%d %d", &a, &b);
	////�������
	//int c = Add(a, b);
	////shuchu
	//printf("%d\n", c);


	//�����Ĳ�����Ϊ ʵ�κ��β�

	//ʵ��



	//return ���
	//return x+y //�������� �ٷ���
	//return ����һ�κ������䲻��ִ��
	//if��֧������������return



	/*int arr[10] = { 1,2,3,4,5,6,7,8,9,10 };
	int sz = sizeof(arr) / sizeof(arr[0]);
	set_arr(arr, sz);
	print_arr(arr, sz);*/


	//zhongdain
	// 1 �β�Ҫ��ʵ�θ���ƥ��
	// 2 ������ʵ�������� �β�Ҳ����д��������ʽ
	// 3 �β������һ������ �����С���Բ�д
	// 4 �β�����Ƕ�ά���� �п���ʡ�� �в���
	// 5 �ββ��������������ʵ����ͬһ������


	//Ƕ�׵��ú���ʽ����
	int y = 0;//��
	int m = 0;//��
	scnaf("%d %d", &y, &m);
	int d = get_days_of_month(y, m);
	return 0;
}