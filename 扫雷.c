#define _CRT_SECURE_NO_WARNINGS

#include "game.h"
void menu()
{
	printf("......................\n");
	printf("..... 1. play.........\n");
	printf("..... 0. exit.........\n");
	printf("......................\n");
}
void game()
{
	//����Ĵ���
	char mine[ROWS][COLS];
	char show[ROWS][COLS];
	InitBoard(mine, ROWS, COLS, '0');
	InitBoard(show, ROWS, COLS, '*');
	
	//���̵Ĵ�ӡ
	 //DisplayBoard(mine, ROW, COL);
	DisplayBoard(show, ROW, COL);
	//������
	SetMine(mine, ROW, COL);
	//DisplayBoard(mine, ROW, COL);
	//�Ų���
	Findmine(mine, show, ROW, COL);
}
int main()
{
	//ɨ����Ϸ˵�� 
	//ʹ�ÿ���̨ʵ�־���ɨ����Ϸ
	//��Ϸ����ͨ���˵�ʵ�ּ���������˳�
	//ɨ�׵�������9*9
	//Ĭ���������10����
	//�����Ų���
	//���λ�ò����� ����ʾ��Χ�м�����
	//���λ�����ף���ը����Ϸ����
	//�ѳ�10����֮��������׶��ҳ��������׳ɹ� ��Ϸ����
	int input = 0;
	srand((unsigned int)time(NULL));
	do
	{
		menu();
		printf("��ѡ��:");
		scanf("%d", &input);
		switch (input)
		{
		case 1:
			game();
			break;
		case 0:
			printf("�˳���Ϸ\n");
			break;
		default:
			printf("ѡ�����������ѡ��");
			break;
		}
	} while (input );
	return 0;
}