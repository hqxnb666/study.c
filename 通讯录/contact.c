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
		printf("通讯录已满，无法增加\n");
		return;
	}
	//增加信息
	printf("请输入名字：");
	scanf("%s\n", pc->data[pc->sz].name);
	printf("请输入年龄：");
	scanf("%d\n", &(pc->data[pc->sz].age));
	printf("请输入性别：");
	scanf("%s", pc->data[pc->sz].sex);
	printf("请输入电话：");
	scanf("%s", pc->data[pc->sz].tele);
	printf("请输入地址：");
	scanf("%s", pc->data[pc->sz].addr);

	pc->sz++;
	printf("增加成功");
}

void ShowContact(Contact* pc)
{
	assert(pc);
	int i = 0;
	//名字 年龄 性别 电话 地址
	printf("%-20s%-5s%-5s%-12s%-30s\n", "名字", "年龄", "性别", "电话", "地址");
	if (pc->sz == 0)
	{
		printf("通讯录为空，无需打印\n");
		return;
	}
	for (i = 0; i < pc->sz; i++)
	{
		//打印每个人信息
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
		printf("通讯录为空 无法删除：");
		return;
	}
	//找到后再删除
	printf("输入要删除人的名字：");
	scanf("%s", name);
	//找到名字为name 的人
	int ret = FindByname(pc, name);
	if (ret == -1)
	{
		printf("要删除的人不存在\n");
		return;
	}
	//删除这个人
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
	printf("请输入查找人名字");
	scanf("%s", name);
	int ret = FindByname(pc, name);
	if (ret == -1)
	{
		printf("要查找的人不存在\n");
		return;
	}
	//显示出来
	printf("%-20s%-5s%-5s%-12s%-30s\n", "名字", "年龄", "性别", "电话", "地址");
	printf("%-20s%-5d%-5s%-12s%-30s\n",
		pc->data[ret].name, pc->data[ret].age, pc->data[ret].sex, pc->data[ret].tele, pc->data[ret].addr);

}

void MODIFYContact(Contact* pc)
{
	char name[NAME_MAX];
	assert(pc);
	printf("请输入要修改的人名字");
	scanf("%s", name);
	int ret = FindByname(pc, name);
	if (ret == -1)
	{
		printf("要修改的人不存在\n");
		return;
	}
	printf("请输入名字：");
	scanf("%s\n", pc->data[ret].name);
	printf("请输入年龄：");
	scanf("%d\n", &(pc->data[ret].age));
	printf("请输入性别：");
	scanf("%s", pc->data[ret].sex);
	printf("请输入电话：");
	scanf("%s", pc->data[ret].tele);
	printf("请输入地址：");
	scanf("%s", pc->data[ret].addr);

	printf("修改成功");
}