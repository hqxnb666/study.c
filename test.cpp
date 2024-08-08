#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
using namespace std;
class Person {
public:
	virtual void BuyTIcket() { 
		cout << "ÂòÆ±È«¼Û" << endl; }
};

class Student : public Person {
public:
	virtual void BuyTIcket() {
		cout << "ÂòÆ±°ë¼Û" << endl;
	}
};
int main()
{
	Person p1;
	Student s1;
	p1.BuyTIcket();
	s1.BuyTIcket();
	return 0;
}