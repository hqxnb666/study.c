#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//int main()
//{
//	int a = 10;
//	int arr[] = { 1,2,3,4,5,6,7 };
//	//c语言给程序员一种权利：能够动态申请管理内存空间
//
//	//malloc 和 free
//	//申请一块空间 存放10个整形
//	//int* p = (int*)malloc(10 * sizeof(int));
//	//如果返回空指针 申请失败
//	int* p = (int*)malloc(INT_MAX*4);
//	if (p == NULL)
//	{
//		perror("malloc");
//		return 1;
//	}
//	 //使用
//	int i = 0;
//	for (i = 0; i < 10; i++)
//	{
//		*(p + i) = i;
//	}
//	free(p);
//	p = NULL;
//
//	//malloc申请的空间怎么释放呢
//	// 1.free释放 -- 主动
//	// 程序退出后  malloc申请的空间也会被操作系统回收的 == 被动
//
//	//正常情况下  谁申请的空间 谁去释放  自己不释放 也要交代别人释放
//	return 0;
//}




//callo
// int main()
//{
	//calloc也是申请空间的
	//calloc(10, sizeof(int)
	//  malloc(10 * sizeof(int));

	//calloc  除了参数的区别 申请号之后 会将空间初始化  但malloc不会
//	int* p = calloc(10 , sizeof(int));
//	if (p == NULL)
//	{
//		perror("malloc");
//		return 1;
//	}
//	int i = 0;
//	for (i = 0; i < 10; i++)
//	{
//		printf("%d\n", *(p + i));
//	}
//	free(p);
//	p = NULL;
//
//	return 0;
//}



//realloc    让动态内存更加灵活
// 如果realloc开辟空间失败  也会返回NULL  所以我们不会直接使用原来的指针接受realloc
//int* ptr = (int*)realloc(p, 20 * sizeof(int));
//if (ptr != NULL)
//{
//	p = ptr;
//}
//free(p);
//p =NULL

//int main()
//{
//	int* p = (int*)realloc(NULL, 40); //  == malloc(40);
//	return 0;
//}



