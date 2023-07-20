//定义变量的方式
//int age = 150
//float weight = 45.5f;
//char ch = 'w'



//变量的分类

//局部变量
//全局变量
#include <stdio.h>
#include "变量.h"
//int a = 20;//全局变量=定义在代码块({})之外的变量
  int Add(int x, int y)
{
	int z = x + y;
	return z;

}
int main_01()

{
	//int a = 10;//局部变量
	//printf("%d\n", a);
	//局部变量和全局变量名字不要一样 会有bug
	//当局部变量和全局变量相同时，局部变量优先




	//计算两个数的和
	int num2 = 10;
    int num1 = 20;
	int sum = 0;
	int a = 100;
	int b = 200;
	sum = Add(num1, num2);
	sum = Add(a, b);
	//输入数据-使用输入函数scanf
//canf_s(&num1, num2);//取地址符号
//nt sum = 0;
	//c语言规定，变量要在当前代码块的最前面
	//m = num1 + num2;
 //ntf("sum = %d\n", sum);



	//extern	int sum;
	//printf("num= %d\n", num);






	

	return 0;
}