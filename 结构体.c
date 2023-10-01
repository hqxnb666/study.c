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
	//话可以ps->age = 18;
	strcpy((*ps).name, "zhangsan");//字符串靠背
}
 void print_s(struct s t)
 {
	 printf("%s %d", t.name, t.age);
 }
int main()
{
	//结构体成员的访问
	//1.   /
	//2.   ->
	//结构体变量.成员
	struct s s = { 0 };
	//写一个函数给s中存放数据
	set_s(&s);
	//写一个函数打印s中的数据
	print_s(s);
	return 0;
}