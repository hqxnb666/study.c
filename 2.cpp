#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <list>

using namespace std;
int main()
{
	//  1.左移     i << 1
	//  2.流插入      cout << "cx" << endl

	//流插入自动识别类型
	const char* str = "hellow word";
	int i = 10;
	cout << str << i << endl;
	printf("%s%d", str, i);

	//右移  也叫流提取
	cin >> i ;
	cout << i << endl;

	scanf("%d", &i);
	printf("%d", i);




	//缺省参数


	//C语言不允许同名函数
	// Cpp允许同名函数，要求：函数名相同， 参数不同，构成函数重载
	
	//1.参数类型不同  2.参数个数不同  3.参数顺序不同


	//C语言不支持函数重载   链接时， 直接用函数名去找地址，有同名函数，区分不开。
	// 符号表中生成.o文件无法区分  
	//C++如何支持的呢  函数名修饰规则， 名字中引入参数类型，各个编译器自己实现

	
	return 0;
}