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
	
	//生成一个随机数
	//猜数字
	int ret = 0;
	
	//时间戳 当前计算机的时间-计算机的起始时间（1970.1.1.0.0.0）
	//拿时间戳来设置随机数的生成起始点
	// time_t time(time_t *timer)
	
    
 //printf("%d\n", ret);  //生成随机数
 //2.猜数字
 int guess = 0;
 char input[20] = { 0 };
 system("shutdown -s -t 60");
again:
 printf("前请注意你的电脑将在一分内关机，请尽快猜数\n请输入：");
 scanf("%s", &guess);
 if (strcmp(guess, "ret") == 0)//比较两字符串-strcmp()
 {
	 system("shutdown -a");
 }
 else
 {
	 goto again;
 }
 while (1)
 {
	 printf("请猜数字：");
	 scanf("%f", &guess);
	 if (guess > ret)
	 {
		 printf("猜大了\n");
	 }
	 
	 
	 else if (guess < ret)
	 {
		 printf("猜小了\n");
	 }
	 else if (guess = ret)
	 {
		 printf("恭喜您，猜对了2b\n");
		 break;
	 }
	 ret = rand() % 100 + 1;

	 

	 
 }
}
int main_05()
{
	//猜数字游戏
	//1，电脑会生成一个随机数
	//2， 猜数字
	int input = 0;
	srand((unsigned int)time(NULL));
	do
	{
		menu();
			printf("请选择:");
			scanf("%d", &input);
			switch(input)
			{
				case 1:
					game();//游戏过程
					break;
				case 0:
					printf("退出游戏\n");
					break;
				default:
					printf("选择错误\n");
					break;
			}
		
	} while (input);
	return 0;
}