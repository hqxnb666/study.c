#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <math.h>
//int main()
//{
//	double r = sqrt(-16.0);
//	printf("%lf\n", r);
//	return 0;
//	 int a;
//}


//�����ķ������������ࣺ
//1.  void - ��ʾ˵����������
//2.  ��������  int char short
//void menu()
//{
//	printf("************************\n");
//	printf("************************\n");
//	printf("************************\n");
//	printf("************************\n");
//	printf("************************\n");
//
//}
//int main()
//{
//	menu();
//}
//int Add(int x, int y) //��ʽ�ϴ��ڣ�����β�
//{
//	
//	return (x+y);
//}
//
//int main()
//{
//	int a = 0;
//	int b = 0;
//	scanf("%d%d", &a, &b);
//	//�������
//	int c = Add(a, b);//a �� b ����ʵ���ݸ�Add�Ĳ�������ʵ�ʲ��������ʵ��
//	//���
//	printf("%d\n", c);
//}

//void test()
//{
//	int n = 0;
//	scanf("%d", &n);
//	printf("hehe\n");
//	if (n == 5)
//		return;
//	printf("haha\n");
//}


//int test1()
//{
//	return 3.14;
//}

//int test2()
//{
//	int n = 0;    //����
//	if (n == 5)
//	{
//		return 1;
//	}
//		//���û��if������ô��
//}
//int main()
//{
//	int n = test2();
//	printf("%d\n", n);
//	return 0;
//}


//void set_arr(int arr[3][5], int r, int l)
//{
//	/*int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		arr[i] = -1;
//	}*/
//}
//void print_arr(int arr[],int  sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%d ", arr[i]);
//	}
//	printf("\n");
//}
//int main()
//{
//	int arr1[10] = { 1,2,3,4,5,6,7,8,9,10 };
//	int arr2[3][5] = { 1,2,3,4,5,2,3,4,5,6,3,4,5,6,7 };
//	int r = 3;
//	int l = 5;
//	int sz = sizeof(arr2) / sizeof(arr2[0]);
//	//��дһ����������arr�е�����ȫ������Ϊ-1;
//	set_arr(arr2,r,l);
//	//дһ����������arr�е����ݴ�ӡ����
//	
//	print_arr2(arr1,sz);
//	return 0;
//}

//int is_leap_year(int y)
//{
//	if ((y % 4 == 0) && (y % 100 != 0) || (y % 400 == 0))
//		return 1;
//	else
//		return 0;
//}
//int get_days_of_month(int y, int m)
//{
//
//	int days[13] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
//	//               0 1   2  3  4 
//	int d = days[m];
//	
//	if (is_leap_year(y) && m == 2)
//	{
//		d += 1;
//	}
//	return d;
//}
////�����ǿ���Ƕ�׵��õģ����ǲ���Ƕ�׶���
//int main()
//{
//	int y = 0; //��
//	int m = 0; //��
//	scanf("%d %d", &y, &m);
//	int d = get_days_of_month(y, m);
//	printf("%d\n", d);
//	return 0;
//}
//#include <string.h>
//int main()
//{
//	printf("%zd\n", strlen("abc"));
//	return 0;
//}



//�����꣬����1
//�������꣬����0
 

int is_leap_year(int y)
{
	if ((y % 4 == 0) && (y % 100 != 0) || (y % 400 == 0))
		return 1;
	else
		return 0;
}

//�����Ķ���Ҳ��һ�����������
int main()
{
	int y = 0;
	scanf("%d", &y);
	if(is_leap_year(y))
	{
		printf("%d ������\n", y);
	}
	else
	{
		printf("%d��������\n", y);
	}

	return 0;
}
//�����Ķ���
