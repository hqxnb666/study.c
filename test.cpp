#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
using namespace std;
class Person {
public:
	 void BuyTIcket() { 
		cout << "买票全价" << endl; }
};

class Student : public Person {
public:
	virtual void BuyTIcket() {
		cout << "买票半价" << endl;
	}
};
//多态的条件
//虚函数的重写（父子类虚函数，要求三同  -- 函数名 参数 返回值）
//                 1.例外（协变）
//                 2.（析构函数）建议析构函数写成虚函数，防止内存泄漏
  
int main()
{
	Person p1;
	Student s1;
	p1.BuyTIcket();
	s1.BuyTIcket();
	return 0;
}