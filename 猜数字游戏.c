#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void menu()
{
	
	
	printf("*****************************\n");
	printf("****   1. play      0.exit***\n");
	printf("*****************************\n");
}
//RAND_MAX 0-32767
void game()
{
	
	//����һ�������
	//������
	int ret = 0;
	
	//ʱ��� ��ǰ�������ʱ��-���������ʼʱ�䣨1970.1.1.0.0.0��
	//��ʱ����������������������ʼ��
	// time_t time(time_t *timer)
	
    
 //printf("%d\n", ret);  //���������
 //2.������
 int guess = 0;
 char input[20] = { 0 };
 system("shutdown -s -t 60");
again:
 printf("ǰ��ע����ĵ��Խ���һ���ڹػ����뾡�����\n�����룺");
 scanf("%s", &guess);
 if (strcmp(guess, "ret") == 0)//�Ƚ����ַ���-strcmp()
 {
	 system("shutdown -a");
 }
 else
 {
	 goto again;
 }
 while (1)
 {
	 printf("������֣�");
	 scanf("%f", &guess);
	 if (guess > ret)
	 {
		 printf("�´���\n");
	 }
	 
	 
	 else if (guess < ret)
	 {
		 printf("��С��\n");
	 }
	 else if (guess = ret)
	 {
		 printf("��ϲ�����¶���2b\n");
		 break;
	 }
	 ret = rand() % 100 + 1;

	 

	 
 }
}
int main_05()
{
	//��������Ϸ
	//1�����Ի�����һ�������
	//2�� ������
	int input = 0;
	srand((unsigned int)time(NULL));
	do
	{
		menu();
			printf("��ѡ��:");
			scanf("%d", &input);
			switch(input)
			{
				case 1:
					game();//��Ϸ����
					break;
				case 0:
					printf("�˳���Ϸ\n");
					break;
				default:
					printf("ѡ�����\n");
					break;
			}
		
	} while (input);
	return 0;
}