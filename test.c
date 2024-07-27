#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
//int main()
//{
//	char ch[] = "abd";
//	scanf("%s", &ch);
//	printf("%s",ch);
//	return 0;
//}

//int main()
//{
//	int a = 0;
//	for (a = 3; a <= 100; a++)
//	{
//		if (a % 3 == 0)
//		{
//			printf("%d ", a);
//		}
//	}
//}

//int main()
//{
//	int a = 0;
//	int b = 0;
//	int c = 0;
//	scanf("%d%d%d", &a, &b, &c);
//	if (a < b)
//	{
//		int tmp = a;
//		a = b;
//		b = tmp;
//	}
//	if (a < c)
//	{
//		int tmp = a;
//		a = c;
//		c = tmp;
//	}
//	if (b < c)
//	{
//		int tmp = b;
//		b = c;
//		c = tmp;
//	}
//	printf("%d %d %d", a, b, c);
//	return 0;
//}


//int main()
//{
//	int i = 0;
//	for (i = 1; i < 9; i++) {
//		int j = 0;
//		for (j = 1; j <= i; j++) {
//			printf("%d*%d=%2d ", i, j, i * j);
//		}
//		printf("\n");
//	}
//	return 0;
//}

//int main()
//{
//	int arr[10] = { 0 };
//	int i = 0;
//	int max = 0;
//	for (int i = 0; i < 10; i++)
//	{
//		scanf("%d", &arr[i]);
//	}
//
//	max = arr[0];
//	for (int i = 1; i < 10; i++)
//	{
//		if (arr[i] > max)
//			max = arr[i];
//	}
//	printf("max = %d", max);
//	return 0;
//}


//int main()
//{
//	int i = 0;
//	double sum = 0.0;
//	int flag = 1;
//	for (int i = 1; i <= 100; i++)
//	{
//		sum += flag*1.0 / i;
//		flag = -flag;
//	}
//	printf("%lf", sum);
//	return 0;
//}

//int main()
//{
//	int i = 0;
//	int count = 0;
//	for (i = 1; i <= 100; i++)
//	{
//		if (i % 10 == 9)
//			count++;
//		if (i / 10 == 9)
//			count++;
//	}
//	count--;
//	printf("%d\n", count);
//}


int main()
{
	int i = 0;
	int count = 0;
	for (i = 101; i <= 199; i++)
	{
		//´Ó2   -  i-1
		int j = 0;
		int flag = 1;
		for (j = 2; j * j<= i; j++)
		{
			if (i % j == 0)
			{
				flag = 0;
				break;
				
			}
		}
		if (flag) {
			count++;
			printf("%d ", i);
		}
	}
	printf("\ncount = %d\n", count);
	return 0;
}