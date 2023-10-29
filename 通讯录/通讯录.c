#define _CRT_SECURE_NO_WARNINGS

#include "contact.h"
void menu()
{
	printf("***********************************\n");
	printf("***1.add           2.del  *********\n");
	printf("***3.search        4.modify *******\n");
	printf("***5. show         6. sort  *******\n");
	printf("***0.exit  ************************\n");
	printf("***********************************\n");
}
enum Option
{
	EXIT,
	ADD,
	DEL,
	SEARCH,
	MODIFY,
	SHOW,
	SORT
};
int main()
{
	//测试通讯录的基本功能
	int input = 0;
	Contact con; //通讯录
	//初始化通讯录
	InitContact(&con);// 一定要传地址 不然会溢出
	do
	{
		menu();
		printf("请输入你的选择:\n");
		scanf("%d\n", &input);
		switch (input)
		{
		case ADD:
			AddContact(&con);
			break;
		case DEL:
			DelContact(&con);
			break;
		case SEARCH:
			SearchContact(&con);
			break;
		case MODIFY:
			MODIFYContact(&con);
			break;
		case SHOW: 
			ShowContact(&con);
			break;
		case SORT:
			break;
		case EXIT:
			printf("推出通讯录\n");
			break;
		default:
			printf("选择错误 请重新选择：\n");
		}
	} while (input);
	return 0;
}
//前面实现的通讯录有什么问题 
//1，通讯录的大小是固定的100个元素 怎么解决 使用动态内存管理