#define _CRT_SECURE_NO_WARNINGS
//  c������һ�� �ṹ���ĳ����������
// 1. ˳��ṹ
// 2. ѡ��ṹ
// 3. ѭ���ṹ
//    ��֧���   ѭ�����   c������һ��;�����ľ���һ�����
#include <stdio.h>
int main_01()
{
	int age = 10;
	if (age < 18)
	{
		printf("δ����\n");

		printf("����̸����\n");
	}//Ҫ��ӡ��������ô����{}
	else if (age >= 18 && age < 28)
		printf("����\n");
	else if (age >= 28 && age < 50)
		printf("׳��");
	else
		printf("�ϲ���\n");





	//switch���
	int day = 0;
	scanf("%d", &day);
	/*if (day == 1)
		printf("����һ\n");
	else if( 2 == day)
		printf("����2\n");*/




	switch (day)//   ���������α��ʽ  case ����������ͳ������ʽ
	{
	case 1:
		printf("����1\n");
		break;
	case 2:
		printf("����2\n");
		break;
	case 3:
		printf("����3\n");
		break;
	case 4:
		printf("����4\n");//break ����ѭ��
		break;
	case 5:
		printf("����5\n");
		break;
	default:
		printf("�������\n");
		break;



		//�����ͬ 1-5������ 6-7��Ϣ��
		//case��һ��Ҫ��
		//  case 1:
		//  case 2:
		//  case 3:
		//  case 4:
		//      printf("������\n");
		//   case 6:
		//   case 7:
		//      printf("��Ϣ��\n");
	}

	//�������̫���� �����Ǿͻ���switch���
	//   switch(���α��ʽ)
	//   {
	//       ����
	//   }
	// 
	//

	return 0;
}
//  if�﷨�ṹ
//   if(���ʽ)
//    ��䣻

//   if���ʽ
//   ���1��
//   else
// ���2��
// 
// 
// ���֧
//   if�����ʽ1��
//       ���1��
//   else if�����ʽ2��
//      ���2��
//  else
/   ���3��
