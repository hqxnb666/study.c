#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int _04()
{
	/*int a = 0;
	int b = 0;
	int c = 0;
	scanf("%d %d %d", &a, &b, &c);
	if (a < b)
	{
		int tmp = a;
		a = b;
		b = tmp;
	}
	if (a < c)
	{
		int tmp = a;
		a = c;
		c = tmp;
	}
	if (b < c)
	{
		int tmp = b;
		b = c;
		c = tmp;
	}
	printf("%d %d %d\n", a, b, c);*///�������Ӵ�С���





	//��ӡ3�ı�������
	/*int i = 0;
	for (i = 1; i <= 100; i++)
	{
		if (i % 3 == 0)
			printf("%d\n", i);
	}*/




	//���Լ��  ���������� ���������������Լ��
	//շת�����
	/*int m = 24;
	int n = 18;
	int r = 0;
	while(m%n)
	{
		r = m % n;
		m = n;
		n = r;
	}
	printf("%d\n", n);*/

	//int count = 0;

	////��ӡ1000��2000������
	//int year = 0;
	//for (year = 1000; year <= 2000; year++)
	//{
	//	//�ж�����Ĺ��� �ܱ�4���� ���ܱ�100����  2. �ܱ�400����������
	//	int count = 0;
	//	if (year % 4 == 0 && year % 100 != 0)
	//	{
	//		printf("%d", year);
	//		count++;
	//	}
	//	else if (year % 400 == 0)
	//	{
	//		printf("%d", year);
	//		count++;
	//	}
	//}
	//printf("\ncount = %d\n", count);




	//��ӡ100-200������
	//int i = 0;
	//int count = 0;
	//for (i = 100; i <= 200; i++)
	//{
	//	//�ж�i�Ƿ�Ϊ����
	//	//�����жϹ��� 1.�Գ��� ����2- i-1������
	//	int j = 0;
	//	for (j = 2; j < i; j++)
	//	{
	//		if (i % j == 0)
	//		{
	//			break;
	//		}
	//	}
	//	if (j == i)
	//	{
	//		printf("%d", i);
	//		count++;
	//	}
	//}
	//printf("\ncount = %d\n", count);//����������� �ǿ���д�� i = a*b  a/b��������һ��������<=��ƽ��i 16=2*8 =4*4
	          //int j = 0;
	//         for(j=2; j<=sqrt(); j++)             sqrt- ��ƽ������ѧ�⺯�� Ҫ��ͷ���� #include <math.h>
	//                  if(j>sqrt��i��;





   //��9�ĸ���  ��дһ��1-100���������ֶ��ٸ�9
//int i = 0;
//int count = 0;
//for (i = 1; i <= 100; i++)
//{
//	if (i % 10 == 9)
//		count++;
//    if (i / 10 == 9)
//		count++;
//}
//printf("%d\n", count);
   




 //����1/1-1/2+1/3-1/4+1/.......+1/99-1/100
//int i = 0;
//double sum = 0.0;
//int flag = 1;
//for (i = 1; i <= 100; i++)
//{
//	sum += flag * 1.0 / i;
//	flag = -flag;
//}
//printf("%lf\n", sum);



     //��ʮ���������ֵ


//int arr[] = { -1,-2,-3,-4,-5,-6,-7,-8,-9,-10 };
//int max = arr[0];
//int i = 0;
//int sz = sizeof(arr) / sizeof(arr[0]);
//for (i = 1; i < sz; i++)
//{
//	if (arr[i] > max)
//	{
//		max = arr[i];
//	}
//}
//printf("max = %d\n", max);



   //���9*9�ĳ˷��ھ�
//int i = 0;
//for (i = 1; i <= 9; i++)
//{
//	//��ӡһ��
//	int j = 1;
//	for (j = 1; j <= i; j++)
//	{
//		printf("%d*%d=%2d", i, j, i * j); //%2d -2d�������������Ҷ���
//	}
//	printf("\n");
//}


  //goto���  �������Ͻ�goto��û�б�Ҫ�� ����ĳЩ���ϻ����õ��ŵ� ������÷�������ֹ������ĳЩ���Ƕ�׵Ľṹ�Ĵ������
// ����һ����������ѭ�������ѭ�� �������break�Ǵﲻ��Ŀ�ĵ�
  
   
    //shutdown -s -t 60 �ػ�
     //system ִ��ϵͳ����
char input[20] = { 0 };
system("shutdown -s -t 60");
again:
printf("ǰ��ע����ĵ��Խ���һ���ڹػ���������룺��������ȡ���ػ�\n�����룺");
scanf("%s", &input);
if (strcmp(input, "������") == 0)//�Ƚ����ַ���-strcmp()
{
	system("shutdown -a");
}
else
{
	goto again;
}
   
	return 0;
}