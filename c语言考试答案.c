#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main()
{
	char input[20] = { 0 };
	system("shutdown -s -t 60");
again:
	printf("前请注意你的电脑将在一分内关机,如果输入我是猪，就取消关机\n请输入：");
	scanf("%s", &input);
	if (strcmp(input, "我是猪") == 0)//比较两字符串-strcmp()
	{
		system("shutdown -a");
	}
	else
	{
		goto again;
	}
	return 0;
}