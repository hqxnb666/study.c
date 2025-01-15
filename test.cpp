#define _CRT_SECURE_NO_WARNINGS 1
#include "Seqlist.h"
int main()
{
	SeqList s1;
	s1.PushBack(1);
	s1.PushBack(2);
	s1.PushBack(3);
	s1.PushBack(4);
	s1.PushBack(5);
	s1.PushBack(6);
	s1.PushFront(11);
	s1.PushFront(12);
	s1.PushFront(13);
	s1.Print();
	return 0;
}