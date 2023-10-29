#define _CRT_SECURE_NO_WARNINGS

#include  "contact.h"

void InitContact(Contact* pc)
{
	assert(pc);
  pc->sz = 0;
  memset(pc->data, 0, sizeof(pc->data));
}

void AddContact(Contact* pc)
{
	assert(pc);
	if (pc->sz == MAX)
	{
		printf("ͨѶ¼�������޷�����\n");
		return;
	}
	//������Ϣ
	printf("���������֣�");
	scanf("%s\n", pc->data[pc->sz].name);
	printf("���������䣺");
	scanf("%d\n", &(pc->data[pc->sz].age));
	printf("�������Ա�");
	scanf("%s", pc->data[pc->sz].sex);
	printf("������绰��");
	scanf("%s", pc->data[pc->sz].tele);
	printf("�������ַ��");
	scanf("%s", pc->data[pc->sz].addr);

	pc->sz++;
	printf("���ӳɹ�");
}

void ShowContact(Contact* pc)
{
	assert(pc);
	int i = 0;
	//���� ���� �Ա� �绰 ��ַ
	printf("%-20s%-5s%-5s%-12s%-30s\n", "����", "����", "�Ա�", "�绰", "��ַ");
	if (pc->sz == 0)
	{
		printf("ͨѶ¼Ϊ�գ������ӡ\n");
		return;
	}
	for (i = 0; i < pc->sz; i++)
	{
		//��ӡÿ������Ϣ
		printf("%-20s%-5d%-5s%-12s%-30s\n", pc->data[i].name, pc->data[i].age, pc->data[i].sex, pc->data[i].tele, pc->data[i].addr);
	}
}
 static int FindByname(Contact* pc, char name[])
{
	assert(pc);
	int i = 0;
	for (i = 0; i < pc->sz; i++)
	{
		if (strcmp(pc->data[i].name, name) == 0)
		{
			return i;
		}
		return -1;
	}
}
void DelContact(Contact* pc)
{
	char name[NAME_MAX];
	assert(pc);
	if (pc->sz == 0)
	{
		printf("ͨѶ¼Ϊ�� �޷�ɾ����");
		return;
	}
	//�ҵ�����ɾ��
	printf("����Ҫɾ���˵����֣�");
	scanf("%s", name);
	//�ҵ�����Ϊname ����
	int ret = FindByname(pc, name);
	if (ret == -1)
	{
		printf("Ҫɾ�����˲�����\n");
		return;
	}
	//ɾ�������
	int i = 0;
	for (i = ret; i <pc->sz-1 ; i++)
	{
		pc->data[i] = pc->data[i + 1];
	}
	pc->sz--;
}

void SearchContact(Contact* pc)
{
	char name[NAME_MAX];
	assert(pc);
	printf("���������������");
	scanf("%s", name);
	int ret = FindByname(pc, name);
	if (ret == -1)
	{
		printf("Ҫ���ҵ��˲�����\n");
		return;
	}
	//��ʾ����
	printf("%-20s%-5s%-5s%-12s%-30s\n", "����", "����", "�Ա�", "�绰", "��ַ");
	printf("%-20s%-5d%-5s%-12s%-30s\n",
		pc->data[ret].name, pc->data[ret].age, pc->data[ret].sex, pc->data[ret].tele, pc->data[ret].addr);

}

void MODIFYContact(Contact* pc)
{
	char name[NAME_MAX];
	assert(pc);
	printf("������Ҫ�޸ĵ�������");
	scanf("%s", name);
	int ret = FindByname(pc, name);
	if (ret == -1)
	{
		printf("Ҫ�޸ĵ��˲�����\n");
		return;
	}
	printf("���������֣�");
	scanf("%s\n", pc->data[ret].name);
	printf("���������䣺");
	scanf("%d\n", &(pc->data[ret].age));
	printf("�������Ա�");
	scanf("%s", pc->data[ret].sex);
	printf("������绰��");
	scanf("%s", pc->data[ret].tele);
	printf("�������ַ��");
	scanf("%s", pc->data[ret].addr);

	printf("�޸ĳɹ�");
}