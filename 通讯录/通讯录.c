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
	//����ͨѶ¼�Ļ�������
	int input = 0;
	Contact con; //ͨѶ¼
	//��ʼ��ͨѶ¼
	InitContact(&con);// һ��Ҫ����ַ ��Ȼ�����
	do
	{
		menu();
		printf("���������ѡ��:\n");
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
			printf("�Ƴ�ͨѶ¼\n");
			break;
		default:
			printf("ѡ����� ������ѡ��\n");
		}
	} while (input);
	return 0;
}
//ǰ��ʵ�ֵ�ͨѶ¼��ʲô���� 
//1��ͨѶ¼�Ĵ�С�ǹ̶���100��Ԫ�� ��ô��� ʹ�ö�̬�ڴ����