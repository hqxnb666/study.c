//#define _CRT_SECURE_NO_WARNINGS//判断数是否为奇数   输出1-100的奇数
//#include <stdio.h>
//#include <Windows.h>
//#include <stdlib.h>
//int main_02()
//{
//	int i = 0;
//	while (i <= 100)
//	{
//		if (i % 2 == 1)
//			printf("%d", i);
//		i++;
//	}//第一种写法
//
//
//
//	int i = 1;
//	while (i <= 100)
//	{
//		printf("%d", i);
//		i + 2;
//	}
//
//
//	//计算n的阶乘
//
//
//	
//	int i = 0;
//	int n = 0;
//	int ret = 1;
//	scanf("%d", &n);
//	for (i = 1; i <= n; i++)
//	{
//		ret = ret * i;
//
//	}
//	printf("%d\n", ret);
//
//
//	//计算1!+ 2! + 3! +...
//	int i = 0;
//	int n = 0;
//	int ret = 1;
//	int sum = 0;
//	
//	for (n = 1; n <= 3; n++)
//	{
//	
//			ret = ret * n;
//		
//	//n的阶乘
//		sum = sum + ret;
//	}
//    printf("sum = %d\n", sum);
//
//
//
//
//	// 3.在一个有序数组中查找具体的某个数字n  编写int binsearch(int x, int v[], int n);功能：在v[0]<=v[1]<=[2]....<=
//	//v[n-1]的数组中查找x
//
//	int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
//	int k = 7;
//	int i = 0;
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	for (i = 0; i < sz; i++)
//	{
//		if (k == arr[i])
//		{
//			printf("找到了，下标是：%d\n", i);
//			break;
//		}
//
//	}
//	if (i = sz)
//		printf("找不到\n");
//
//
//
//	//4.编写代码 演示多个字符从两端移动，向中间汇聚
//	
//	char arr1[] = "welcome to bit!!!!!";
//	char arr2[] = "###################";
//	int left = 0;
//	int right = strlen(arr1)-1;
//	while (left<=right)
//	{
//		arr2[left] = arr1[left];
//		arr2[right] = arr1[right];
//		printf("%s\n", arr2);
//		//休息一秒 引用头文件
//		Sleep(1000);
//		system("cls");//cls 清空屏幕 引用头文件
//		left++;
//		right--;
//	}
//	printf("%s\n", arr2);
//
//
//
//
//	//5.编写代码实现，模拟用户登陆情景，并且只能登陆三次 （只允许输入三次密码，如果正确则提示成功，如果三次均错误，则退出程序
//
//	int i = 0;
//	char password[20] = { 0 };
//	for (i = 0; i < 3; i++)
//	{
//		printf("请输入密码:");
//		scanf("%s", password);
//		if (strcmp(password, "123456") == 0)// ==不能用来比较两个字符串相等  应该用一个库函数 strcmp
//		{
//			printf("登录成功\n");
//			break;
//		}
//		else
//			printf("密码错误\n");
//	}
//	if (i == 3)
//		printf("三次密码均错误，退出程序\n");
//	return 0;
//}