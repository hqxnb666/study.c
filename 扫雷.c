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
	//数组的创建
	char mine[ROWS][COLS];
	char show[ROWS][COLS];
	InitBoard(mine, ROWS, COLS, '0');
	InitBoard(show, ROWS, COLS, '*');
	
	//棋盘的打印
	 //DisplayBoard(mine, ROW, COL);
	DisplayBoard(show, ROW, COL);
	//布置雷
	SetMine(mine, ROW, COL);
	//DisplayBoard(mine, ROW, COL);
	//排查雷
	Findmine(mine, show, ROW, COL);
}
int main()
{
	//扫雷游戏说明 
	//使用控制台实现经典扫雷游戏
	//游戏可以通过菜单实现继续玩或者退出
	//扫雷的棋盘是9*9
	//默认随机布置10个雷
	//可以排查雷
	//如果位置不是雷 就显示周围有几个雷
	//如果位置是雷，就炸死游戏结束
	//把除10个雷之外的所有雷都找出来，排雷成功 游戏结束
	int input = 0;
	srand((unsigned int)time(NULL));
	do
	{
		menu();
		printf("请选择:");
		scanf("%d", &input);
		switch (input)
		{
		case 1:
			game();
			break;
		case 0:
			printf("退出游戏\n");
			break;
		default:
			printf("选择错误，请重新选择");
			break;
		}
	} while (input );
	return 0;
}