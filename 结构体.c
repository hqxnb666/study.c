#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
struct s
{
	char name[20];
	int age;
};
 void set_s(struct s* ps)
{
	(*ps).age = 10;
	//������ps->age = 18;
	strcpy((*ps).name, "zhangsan");//�ַ�������
}
 void print_s(struct s t)
 {
	 printf("%s %d", t.name, t.age);
 }
int main()
{
	//�ṹ���Ա�ķ���
	//1.   /
	//2.   ->
	//�ṹ�����.��Ա
	struct s s = { 0 };
	//дһ��������s�д������
	set_s(&s);
	//дһ��������ӡs�е�����
	print_s(s);
	return 0;
}