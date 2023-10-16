#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
//int Add(int x, int y)
//{
//	return x + y;
//}
//int Sub(int x, int y)
//{
//	return x - y;
//}
//int Mul(int x, int y)
//{
//	return x * y;
//}
//int Div(int x, int y)
//{
//	return x / y;
//}
//void menu()
//{
//	printf("*****1.add*******\n");
//	printf("*****2.sub*******\n");
//	printf("*****3.mul********\n");
//	printf("*****4.div********\n");
//	printf("*****0.exit*******\n");
//		
//}
/*nt  Add(int x, int y)
{
	return x + y;

}*/
 /*void bubble_sort(int arr[], int sz)
{
	int i = 0;
	int temp = 0;

	for (i = 0; i < sz - 1; i++);
	{
		for (int j = 0; j <sz-1-i; j++)
		{
			if (arr[j] > arr[j + 1])
			{
				temp = arr[j];
				arr[j] = arr[j+1];
				arr[j + 1] = temp;

			}
		}
	}
}*/
int main()
{
	////函数指针-指向函数的指针 存放的是函数的地址
	////&函数名 - 函数的地址
	////函数名 - 函数的地址
	//printf("%p", &Add);
 //  int	(*pf)(int, int) = &Add;//pf就是一个函数指针变量
 //  int arr[10];
 //  int(*pa)[10] = &arr;//pa是一个数组指针变量


	//int (*pf1)(int, int) = &Add;
	//int (*pf2)(int, int) = &Sub;

	//int (*pfArr[4])(int, int) = { &Add, &Sub }; //存放函数指针的数组
	//数组中存放类型相同的多个元素
	// 
	// 
	//函数指针数组
	
	/*int input;
	int x, y;
	int ret = 0;*/
	//以下是笨拙写法
	/*do
	{
		menu();
		printf("请选择：");
		scanf("%d", &input);
		switch (input)
		{
		case 1:
			printf("请输入2个操作数：");
			scanf("%d %d", &x, &y);
		ret = Add(x, y);
		printf("%d\n", ret);
		
			break;
		case 2:
			printf("请输入2个操作数：");
			scanf("%d %d", &x, &y);
			ret = Sub(x, y);
			printf("%d\n", ret);

			break;
		case 3:
			printf("请输入2个操作数：");
			scanf("%d %d", &x, &y);
			ret = Mul(x, y);
			printf("%d\n", ret);

			break;
		case 4:
			printf("请输入2个操作数：");
			scanf("%d %d", &x, &y);
			ret = Div(x, y);
			printf("%d\n", ret);

			break;
		case 0:
			printf("推出计算器");
			break;
		default:
			printf("选择错误，重新选择：\n");
			break;

		}
	} while (input);*/

	//以下是巧妙写法
	
	//do
	//{
	//	menu();
	//	printf("请选择：\n");
	//	scanf("%d", &input);
	//	//函数指针数组
	//	int(*pfArr[])(int, int) = {  NULL, Add,Sub,Mul,Div };
	//	//                         0     1   2   3   4
	//	if (0 == input)
	//	{
	//		printf("退出\n");
	//	}
	//	else if (input >= 1 && input <= 4)
	//	{
	//		printf("请输入2个操作数\n");
	//		scanf("%d %d", &x, &y);
	//		ret = pfArr[input](x, y);
	//		printf("%d\n", ret);
	//	}
	//	else
	//	{
	//		printf("选择错误，重新选择\n");
	//	}
	//} while (input);


	//指向函数指针数组的指针
	//int (*pfArr[5])(int, int) = { 1,2,3,4,5 }; //pfArr是函数指针数组

	//int (*(*p)[5])(int, int) = &pfArr;




     //回调函数  一个通过函数指针调用的函数 
   

    //qsort -一个库函数
//底层使用的快速排序的方式 对数就进行排序
//这个函数可以直接使用，这个函数可以用来排序任意类型的数据
   


//冒泡排序
//int arr[] = { 1,2,3,4,5, };
//int sz = sizeof(arr) / sizeof(arr[0]);
//bubble_sort(arr,sz);




void qsort(void* base, //待排序数组的第一个元素的地址
	size_t num, //待排序的元素个数
	size_t size,//待排序数组中第一个元素的大小
	int(*compar)(const void*, const void*));//函数指针  1.整型数组  2.排序结构体数组  1.可以直接使用>比较
//2.不能直接使用

//void*的指针不能解引用也不可以加减
//只能存放任意类型数据的地址
int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
qsort(arr,sz，sizeof(arr)[0],)




struct Stu
{
	char name[20];
	int age;
};
int cmp(const void* e1, const void* e2)
{
	((struct Stu*)e1)->age - ((struct Stu*)e2)->age;
}
int cmp_name(const void* e1, const void* e2)
{
	return strcmp(((struct Stu*)e1)->name , ((struct Stu*)e2)->name);
}
void test()
{
	struct Stu arr[] = { {"zhangsan",20},{"lisi",30} };
}
 return 0;
}