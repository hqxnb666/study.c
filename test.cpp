#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
using namespace std;
class Person {
public:
	 void BuyTIcket() { 
		cout << "��Ʊȫ��" << endl; }
};

class Student : public Person {
public:
	virtual void BuyTIcket() {
		cout << "��Ʊ���" << endl;
	}
};
//��̬������
//�麯������д���������麯����Ҫ����ͬ  -- ������ ���� ����ֵ��
//                 1.���⣨Э�䣩
//                 2.������������������������д���麯������ֹ�ڴ�й©
  
int main()
{
	Person p1;
	Student s1;
	p1.BuyTIcket();
	s1.BuyTIcket();
	return 0;
}