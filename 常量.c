//���泣��
//const���εĳ�����
//#define����ı�ʶ������
//ö�ٳ���



#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main_02()
{
	/*int num = 4;
	3;
	100;//���泣��*/


    //const - ������
	//const���εĳ�����   num ���ܸ���
	//const int num = 4;
	//printf("%d\n", num);
	//num = 8;
	//printf("%d\n", num);





     //const	int n = 10;//n�Ǳ����������г����ԣ���������˵n�ǳ�����
	//int arr[n] = { 0 };





	//#define����ı�ʶ������
   // #define MAX 10

	//int arr[MAX] = { 0 };
	//printf("%d\n", MAX);//��������10





	//ö�ٳ���--һһ�о�
	//�Ա��У� Ů������
	//��ԭɫ���� �� ��
	
	//ö�ٹؼ���--enum

	enum Sex
	{
		MALE,
		FEMELE,
		SECRET,


	};
	
	printf("%d\n", MALE);//0
	printf("%d\n", FEMELE);//1
	printf("%d\n", SECRET);//2
	//ö�ٳ������ɸ���

	return 0;
}