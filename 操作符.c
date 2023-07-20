//  算术操作符  + - * / %
//  移位操作符 <<  >>
//  位操作符  $ ^ |

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main_07()
{
	//int a = 5 % 2;//1 取模
	//printf("%d\n", a);




	//<< 左移 移（2进制）位操作符
	//>> 右移
	/*int a = 1;*/
	//整形1占4个字节-32个比特位
	//00000000000000000000000000000000000000000001
	/*int b = a << 1;
	printf("%d\n", b);*/





	//2进制位操作符
	//& 按位于
	// | 按位或
	// ^ 按位异或
	//int a = 3;// 011    二进制1    1            1         1
	//int b = 5;//101d    1*2^3     1*2^2      1*2^1      1*2^0
	//                   8        4           2           1
	//              3      0       0          1          1
	//              5     0      1            0           1\
	//         异或的计算规律
	// 对应的2进制相同为0
	// 对应的2进制相异为1
	//011
	//101
	//110
	  //int c = a^b;
	 // printf("%d\n", c);








	//int a = 10;
	//a = 20;//= 赋值  == 判断相等
	//a = a + 10;
	//a += 10;//完全相等



	//单目操作符
	//双目操作符
	//三木操作符
	//int a = 10;
	//int b = 20;
	//a + b;//双目操作符
	//c语言中我们表示真假  0-假  非0-真
	//int a = 10;
	//printf("%d\n", a);
	//printf("%d\n", !a);  //！  逻辑反操作 真改假 假改真




	/*int a = -2;
	int b = -a;
	int c = +3;*///正号会省略



	//int a = 10;
	//printf("%d\n", sizeof(a));//4 计算字节大小
	//printf("%d\n", sizeof a);








	//int arr[10] = { 0 };
	//int sz = 0;//10个整形元素的数组
	//10*sizeof (int) = 40
	//printf("%d\n", sizeof(arr));
	//计算数组元素个数
	//数组总大小/每个元素大小
	// sz = sizeof(arr) / sizeof(arr[0]);
	// printf("%d\n", sz);








	//int a = 0;//4个字节 32个比特位
	//int b = ~a;
	//printf("%d\n", b);
	//~--按位取反
	//000000000000000000000000000000
	//111111111111111111111111111111





//int a = 10;
//int b = a++;//后置++,先使用, 再使用++// 前置先++
//printf("a = %d\n b = %d\n", a, b);
   


  









  //真 - 非0
  //假 - 0
  //&& - 逻辑与
//int a = 3;
//int b = 5;
//   int c = a && b;
//   printf("c = %d\n", c);


 
 


  //|| - 逻辑或  有一个为真则为真


int a = 10;
int b = 20;
int max = 0;
max = (a > b ? a : b);//   条件操作符 a>b为真， 取a  若为假 取 b 也叫三目操作符
if (a > b)
max = a;
else
max = b;




  //逗号表达式
  
  //下标
//int Add(int x, int y)
//{
//	int z = 0;
//	z = x + y;
//	return z;
//
//}
//int arr[10] = { 0 };
// int (arr)[4]; //[]下标引用操作符
//int a = 10;
//int b = 10;
//int Add(a, b);
//int sum = Add(a, b);//()函数调用操作符




  

	return 0;
}
