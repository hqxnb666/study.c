#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
struct Book
{
	char name[20];
	short price;//55

};//  这个分号必不可缺
int ma1in()
{
    struct Book b1 = { "Cyuyan", 55};
	 struct Book* pb = &b1;
	 printf("%s\n", pb->name);
	 printf("%d\n", pb->price);
	 //  . 结构体变量。成员
	 //-> 结构体指针->成员



	 //name是一个地址
	 strcpy(b1.name, "C++");//strcpy-string copy-字符串拷贝-库函数-string.h
	 printf("%s\n", b1.name);



	 //如何利用pb打印署名价格
	 printf("%s\n", (*pb).name);
	 printf("%d\n", (*pb).price);
	printf("书名:%s\n", b1.name);
	printf("价格:%d\n", b1.price);
	b1.price = 15;
	printf("修改后的价格:%d\n", b1.price);S
	//利用结构体类型-创建一个该类型的结构体变量



	return 0;
	//结构体  人 - 书 复杂对象
	//名字 身高等等
	//复杂对象==结构体- 我们自己创造出来的一种类型
	
}