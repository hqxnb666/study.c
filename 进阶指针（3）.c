//#define _CRT_SECURE_NO_WARNINGS 1
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//void swap(void* buf1, void* buf2, size_t size)
//{
//	int i = 0;
//	for (i = 0; i < size; i++)
//	{
//		char tmp = *((char*)buf1 + i);
//		*((char*)buf1 + i) = *((char*)buf2 + i);
//		*((char*)buf2 + i) = tmp;
//	}
//}
//void pprint(struct Stu arr[], int sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%s", arr[i].name);
//	}
//	printf("\n");
//}
//void bubble_sort(void* base, size_t num,size_t size,int(*cmp)(const void*e1, const void* e2))
//{
//	//冒泡排序堂叔
//	int i = 0;
//	for (i = 0; i <num-1; i++)
//	{
//		int j = 0;
//		for (j = 0; j < num - i - 1; j++)
//		{
//			if (cmp((char*)base + j * size, (char*)base + (j + 1) * size)> 0)
//			{
//				//交换
//				swap((char*)base + j * size, (char*)base + (j + 1) * size, size);
//			}
//		}
//	}
//}
//
//struct Stu
//{
//	char name[20];
//	int age;
//};
//
//int cmp(const void* e1, const void* e2)
//{
//	return *(int*)e1 - *(int*)e2;
//}
//void test1()
//{
//	int arr[] = { 6,7,8,9,10,2,3,4,5 };
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	bubble_sort(arr, sz, sizeof(arr[0]), cmp);
//	pprint(arr, sz);
//}
//int cmp_age(const void* e1, const void* e2)
//{
//	return ((struct Stu*)e1)->age - ((struct Stu*)e2)->age;
//}
//int cmp_name(const void* e1, const void* e2)
//{
//	return strcmp(((struct Stu*)e1)->name , ((struct Stu*)e2)->name);
//}
//void test2()
//{
//	struct Stu arr[] = { {"zhangsan",20}, {"lisi",30},{"wangwu",60} };
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	bubble_sort(arr, sz, sizeof(arr[0]), cmp_name);
//	pprint(arr, sz);
//}
//int main()
//{
//	//冒泡排序 实现一个排序函数
//	//test1();//测试整形
//	test2();//测试结构
//	
//	return 0;
//}





//#define _CRT_SECURE_NO_WARNINGS 1
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//
//
//// 声明 Stu 结构体
//struct Stu
//{
//    char name[20];
//    int age;
//};
//
//// 交换函数
//void swap(void* buf1, void* buf2, size_t size)
//{
//    int i;
//    for (i = 0; i < size; i++)
//    {
//        char tmp = *((char*)buf1 + i);
//        *((char*)buf1 + i) = *((char*)buf2 + i);
//        *((char*)buf2 + i) = tmp;
//    }
//}
//
//// 打印结构体数组
//void pprint(struct Stu arr[], int sz)
//{
//    int i;
//    for (i = 0; i < sz; i++)
//    {
//        printf("%s\n", arr[i].name);
//    }
//}
//
//// 冒泡排序
//void bubble_sort(void* base, size_t num, size_t size, int(*cmp)(const void* e1, const void* e2))
//{
//    int i, j;
//    for (i = 0; i < num - 1; i++)
//    {
//        for (j = 0; j < num - i - 1; j++)
//        {
//            if (cmp((char*)base + j * size, (char*)base + (j + 1) * size) > 0)
//            {
//                // 交换
//                swap((char*)base + j * size, (char*)base + (j + 1) * size, size);
//            }
//        }
//    }
//}
//
//// 比较函数按年龄排序
//int cmp_age(const void* e1, const void* e2)
//{
//    return ((struct Stu*)e1)->age - ((struct Stu*)e2)->age;
//}
//
//// 比较函数按名字排序
//int cmp_name(const void* e1, const void* e2)
//{
//    return strcmp(((struct Stu*)e1)->name, ((struct Stu*)e2)->name);
//}
//
//void test1()
//{
//    int arr[] = { 6, 7, 8, 9, 10, 2, 3, 4, 5 };
//    int sz = sizeof(arr) / sizeof(arr[0]);
//    bubble_sort(arr, sz, sizeof(arr[0]), cmp_age);
//    // 使用正确的函数打印整数数组
//    for (int i = 0; i < sz; i++) {
//        printf("%d ", arr[i]);
//    }
//    printf("\n");
//}
//
//void test2()
//{
//    struct Stu arr[] = { {"zhangsan", 20}, {"lisi", 30}, {"wangwu", 60} };
//    int sz = sizeof(arr) / sizeof(arr[0]);
//    bubble_sort(arr, sz, sizeof(arr[0]), cmp_name);
//    pprint(arr, sz);
//}
//
//int main()
//{
//    // 调用测试函数
//    test1(); // 测试整数
//    test2(); // 测试结构体
//
//    return 0;
//}





//#include <stdio.h>
//int main()
//{
//	int a[] = { 1,2,3,4 };
//
//	printf("%d\n", sizeof(a));//16  数组名a单独放在sizeof内部，数组名表示整个数组，计算的是整个数组单位的字节是16字节
//
//	printf("%d\n", sizeof(a + 0));//a并非到哪都放在sizeof内部 也没有& 所以数组名是首元素地址 a+0 还是首元素地址
//	//是地址就是4/8 byte
//	printf("%d\n", sizeof(*a)); //a是首元素地址 *a就是首元素  就是4Byte  
//	//    *a == *(a+0) == a[0]
//	printf("%d\n", sizeof(a+1));//a并非到哪都放在sizeof内部 也没有& 所以数组名是首元素地址 a+1就是第二个元素的地址
//	//a+1 == &a[1]  是第二个元素的地址 是地址就是4/8个字节
//
//	printf("%d\n", sizeof(a[1]));//4
//	printf("%d\n", sizeof(&a)); //是取数组的地址 但是数组的地址也是地址 地址就是4/8
//	//数组的地址 和 数组首元素的地址 本质区别的类型的区别
//	//a  --  int*    int * p = a
//	//&a  -- int (*)[4]    int (*p)[4] = &a
//	printf("%d\n", sizeof(*&a)); //16  取的是数组& *相互抵消
//	printf("%d\n", sizeof(&a + 1));//4/8  还是地址 
//	printf("%d\n", sizeof(&a[0]));//是首元素地址 计算的是地址的大小 4/8
//	printf("%d\n", sizeof(&a[0] + 1));//4/8
//	return 0;
//}


//#include <stdio.h>
//
//int main()
//{
//	char arr[] = { 'a', 'b', 'c', 'd', 'e', 'f' };
//
//	printf("%d\n", sizeof(arr));// 6   数组名ar单独放在sizeof内部计算的是整个数组的大小 
//
//	printf("%d\n", sizeof(arr + 0));// 4/8   arr是首元素地址 == &arr[0]
//	printf("%d\n", sizeof(*arr));//   arr是首元素地址 *arr就是首元素 1
//	printf("%d\n", sizeof(arr[1]));//;1
//	printf("%d\n", sizeof(&arr));//4/8
//	printf("%d\n", sizeof(&arr + 1));//4/8
//	printf("%d\n", sizeof(&arr[0] + 1));//4/8
//
//	//strlen求字符串长度
//	//统计的是\0出现之前的字符串长度
//	printf("%d\n", strlen(arr));//随机值  arr是首元素地址
//	printf("%d\n", strlen (arr));//
//	printf("%d\n", strlen(arr + 0));//arr+0还是首元素地址
//	printf("%d\n", strlen(*arr));//首元素  == ‘a’-97  
//	//站在strlen角度 传参进去的'a'-97就是地址 ，97作为地址 直接进行访问就是非法访问
//	printf("%d\n", strlen(arr[1]));//‘b’-98 同样是个错误的代码
//	printf("%d\n", strlen(&arr));//&arr -- char(*)[6]
//	//const char*   是个随机值
//	printf("%d\n", strlen(& arr + 1));//也是个随机值
//	printf("%d\n", strlen(&arr[0] + 1));//也是个随机值
//
//	return 0;
//}



#include <stdio.h>
int main()
{

	return 0;
}