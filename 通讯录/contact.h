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
	
	PeoInfo data[MAX]; //���data�������������100���˵���Ϣ
	//һ��ʼû����Ϣ 
	int sz;//��¼��ǰ����
}Contact;

//��ʼ��ͨѶ¼
void InitContact(Contact* pc);
//������ϵ��
void AddContact(Contact* pc);



//��ʾ������ϵ��
void ShowContact(Contact* pc);
//ɾ����ϵ��
void DelContact(Contact* pc);

void SearchContact(Contact* pc);
void MODIFYContact(Contact* pc);