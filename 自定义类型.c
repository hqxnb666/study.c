#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
//struct S1
//{
//	char c1;
//	char c2;
//	int i;
//};
//int main()
//{
//	//结构体内存对齐
//	printf("%d\n", sizeof(struct S1));
//	return 0;
//}
// 
// 
// 
//offsetof  测试确实开辟了这些空间
//对齐规则
//结构体第一个变量在内存中偏移量为0
// 其他成员变量要对其到某个数字 的整数倍地址

//对其数 = 编译器默认的一个对其数 与该成员大小的较小值
//VS中默认值为8
//linux中没有默认对其书 就是自身成员大小


//为什么要对其
 
//1.平台原因  不是所有硬件平台都能访问任意地址的数据
//2.性能原因  为了访问未对其的内存 处理其需要两次内存访问  对其的只需要一次

//尽量让占用空间小的集中在一起




//修改默认对其数
//#pragma pack(1)
//struct Stu
//{
//	char c2;
//};
//#pragma pack()




//位段

//结构体实现位段的能力
//设计之初就是为了节省空间
//位段的成员必须是  int unsigned int  singned int
//位段的成员名后边有一个冒号和一个数字


//位段 的位---是二进制位
//  -a占用两个比特位的空间
//  -b占用5个比特位的空间
//struct A
//{
//	int _a:2;
//	int _b:5;
//	int _c;
//	int _d;
//};


//位段的内存分配
//  char int unsinged int
//  基本按照需要4个字节 或者（char）一个字节来分配
//  涉及很多不确定因素  注重可移植的程序避免使用位段



//位段的跨平台问题
// int 被当成有符号还是无符号不确定
//  位段的最大位数目不能确定
//位段成员内存从左到右或者从右到左尚未定义
//  当一个结构两个位段  第二个位段成员较大 无法容纳于第一个位段的位时 舍弃还是另开辟一个 这是不确定的




//枚举--一一列举
//enum Sex
//{
//	//枚举的可能取值
//	MALE,//枚举常量
//	FEMALE,
//	SECRET
//};
//enum Color
//{
//	RED,
//	GREEN,
//	BLUE
//};
//int main()
//{
//	printf("%d", MALE);
//	printf("%d", FEMALE);//依次递增
//	return 0;
//}




//联合（共用体）也是自定义类型
//union

//union Un
//{
//	char c;
//	int i;
//};
//int main()
//{
//	union Un un;
//	printf("%d\n", sizeof(un));
//	printf("%p\n", &un);
//	printf("%p\n", &(un.c));
//	printf("%p\n",&(un.i));    //公用一个空间
//	return 0;
//}




int check_sys()
{
	union Un
	{
		char c;
		int i;
	}u;
	u.i = 1;
	return u.c;
}

int main()
{
	int ret = check_sys();
	if (ret == 1)
		printf("小端\n");
	else
		printf("大端\n");
	return 0;
}



//联合体大小的计算
//至少是最大成员的大小
//最大成员大小不是对其数整数倍 就扩大到整数倍
