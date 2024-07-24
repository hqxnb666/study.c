#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>

//main - 固定的名字
//主函数 是程序的入口

//main函数是必须有的，但是有且只有一个

//我们C语言中所有的字符必须是英文字符

//int 代表整形  说明Main函数返回一个整形

//int main()
//{
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	return 0;
//}
//
//
//int main()
//{
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	return 0;
//}



//void main()
//{
//	printf("hh\n");
//}

//int main(void)
//{
//
//}


//int main(int argc, char* argv[])
//{
//	/...
//
//}


//int main() {
//
//	//写代码的地方
//	printf("hello word\n");
//
//	//printf -- 库函数 -C语言标准库提供的一个现成的函数-是可以直接使用的
//	//功能是在屏幕上打印信息
//	//库函数。我们必须包含他对应的头文件。 stdio.h
//
//
//
//	return 0;
//}

//%d -十进制的形式来打印整数
//int main()
//{
//	printf("%d\n", sizeof(char));
//	printf("%d\n", sizeof(short));
//	printf("%d\n", sizeof(int));
//	printf("%d\n", sizeof(long));
//	printf("%d\n", sizeof(long long));
//	printf("%d\n", sizeof(double));
//	printf("%d\n", sizeof(float));
//
//	return 0;
//}



//全局变量： 在{}外边定义的变量就是全局变量
//局部变量： 在{}内部定义的变量就是局部变量
//当前局部和全局变量在一个地方都可以使用的同时，局部优先
//int a = 10;
//
//int main()
//{
//	int a = 1000;
//	{
//		int b = 0;
//		printf("%d", b);
//	}
//	//printf("%d", a);
//	return 0;
//}


int main()
{
	int a = 0;
	int b = 0;
	int s = 0;
	//输入2个数
	scanf("%d %d", &a, &b);
	//计算
	s = a + b;
	//输出
	printf("%d\n", s);

	return 0;
}