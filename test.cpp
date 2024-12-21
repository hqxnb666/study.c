#define _CRT_SECURE_NO_WARNINGS 1
//#include <iostream>
#include "AVLTree.h"
//using namespace std;
int main()
{
	int a[] = { 1,3,4,7,8,6,10 };
	AVLTree<int, int> a1;
	for (auto e : a)
	{
		a1.insert({ e,e });
	}
	
	a1.InOrder();
	cout << a1.IsBalance() << endl;
	return 0;
}