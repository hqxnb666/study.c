#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
struct Book
{
	char name[20];
	short price;//55

};//  ����ֺűز���ȱ
int ma1in()
{
    struct Book b1 = { "Cyuyan", 55};
	 struct Book* pb = &b1;
	 printf("%s\n", pb->name);
	 printf("%d\n", pb->price);
	 //  . �ṹ���������Ա
	 //-> �ṹ��ָ��->��Ա



	 //name��һ����ַ
	 strcpy(b1.name, "C++");//strcpy-string copy-�ַ�������-�⺯��-string.h
	 printf("%s\n", b1.name);



	 //�������pb��ӡ�����۸�
	 printf("%s\n", (*pb).name);
	 printf("%d\n", (*pb).price);
	printf("����:%s\n", b1.name);
	printf("�۸�:%d\n", b1.price);
	b1.price = 15;
	printf("�޸ĺ�ļ۸�:%d\n", b1.price);S
	//���ýṹ������-����һ�������͵Ľṹ�����



	return 0;
	//�ṹ��  �� - �� ���Ӷ���
	//���� ��ߵȵ�
	//���Ӷ���==�ṹ��- �����Լ����������һ������
	
}