#define _CRT_SECURE_NO_WARNINGS
#define NAME_MAX 20
#define SEX_MAX 5
#define TELE_MAX 12
#define ADDR_MAX 20
#define MAX 100
#include <stdio.h>
#include <assert.h>
#include <string.h>
typedef struct PeoInfo
{
	char name[NAME_MAX];
	char sex[SEX_MAX];
	char tele[TELE_MAX];
	char addr[ADDR_MAX];
	int age;
}PeoInfo;

typedef struct Contact
{
	
	PeoInfo data[MAX]; //这个data数组有能力存放100个人的信息
	//一开始没有信息 
	int sz;//记录当前人数
}Contact;

//初始化通讯录
void InitContact(Contact* pc);
//增加联系人
void AddContact(Contact* pc);



//显示所有联系人
void ShowContact(Contact* pc);
//删除联系人
void DelContact(Contact* pc);

void SearchContact(Contact* pc);
void MODIFYContact(Contact* pc);