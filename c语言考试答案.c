#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main()
{
	char input[20] = { 0 };
	system("shutdown -s -t 60");
again:
	printf("ǰ��ע����ĵ��Խ���һ���ڹػ�,���������������ȡ���ػ�\n�����룺");
	scanf("%s", &input);
	if (strcmp(input, "������") == 0)//�Ƚ����ַ���-strcmp()
	{
		system("shutdown -a");
	}
	else
	{
		goto again;
	}
	return 0;
}