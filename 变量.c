//��������ķ�ʽ
//int age = 150
//float weight = 45.5f;
//char ch = 'w'



//�����ķ���

//�ֲ�����
//ȫ�ֱ���
#include <stdio.h>
#include "����.h"
//int a = 20;//ȫ�ֱ���=�����ڴ����({})֮��ı���
  int Add(int x, int y)
{
	int z = x + y;
	return z;

}
int main_01()

{
	//int a = 10;//�ֲ�����
	//printf("%d\n", a);
	//�ֲ�������ȫ�ֱ������ֲ�Ҫһ�� ����bug
	//���ֲ�������ȫ�ֱ�����ͬʱ���ֲ���������




	//�����������ĺ�
	int num2 = 10;
    int num1 = 20;
	int sum = 0;
	int a = 100;
	int b = 200;
	sum = Add(num1, num2);
	sum = Add(a, b);
	//��������-ʹ�����뺯��scanf
//canf_s(&num1, num2);//ȡ��ַ����
//nt sum = 0;
	//c���Թ涨������Ҫ�ڵ�ǰ��������ǰ��
	//m = num1 + num2;
 //ntf("sum = %d\n", sum);



	//extern	int sum;
	//printf("num= %d\n", num);






	

	return 0;
}