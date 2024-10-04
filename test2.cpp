#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
using namespace std;
//抽象类
class Car
{
public:
	//纯虚函数
	virtual void Drive() = 0;
	//无法实例化

};
int main()
{
	//Car c;
	
	return 0;
}