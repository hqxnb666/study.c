#define _CRT_SECURE_NO_WARNINGS 1
#include "Seqlist.h"
#include <stdlib.h>
#include <assert.h>
SeqList::SeqList() : a(nullptr), size(0),capacity(0){}

SeqList::~SeqList()
{
	if (a) delete[] a;
	a = nullptr;
	this->size = 0;
	this->capacity = 0;
}
void SeqList::Print() 
{
	assert(a);
	for (int i = 0; i < size; i++)
	{
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}
void SeqList::CheckCapacity()
{
	assert(a);
	if (size == capacity) {
		int newcapacity = capacity == 0 ? 4 : capacity * 2;
		SLDataType* tmp = new SLDataType[newcapacity];
		assert(tmp);
		for (int i = 0; i < size; i++)
		{
			tmp[i] = a[i];
		}
		delete[] a;
		a = tmp;
		capacity = newcapacity;
	}
}

void SeqList::PushBack(SLDataType x)
{
	assert(a);
	CheckCapacity();
	a[size] = x;
	size++;
}

void SeqList::PushFront(SLDataType x)
{
	assert(a);
	CheckCapacity();
	int end = --size;
	for (end; end >= 0; end--)
	{
		a[end + 1] = a[end];
	}
	a[0] = x;
	size++;
}
void SeqList::PopBack()
{
	assert(a);
	assert(size);
	size--;
}
void SeqList::PopFront()
{
	assert(a);
	int begin = 1;
	while (begin < size)
	{
		a[begin - 1] = a[begin];
		begin++;
	}
	size--;
}
void SeqList::Insert(int pos, SLDataType x)
{
	assert(a);
	CheckCapacity();
	int end = size - 1;
	while (end >= pos) {
		a[end + 1] = a[end];
		end--;
	}
	a[pos] = x;
	size++;
}
void SeqList::Erase(int pos, SLDataType x)
{
	assert(a);
	CheckCapacity();
	int begin = pos + 1;
	while (begin < size)
	{
		a[begin - 1] = a[begin];
		begin++;
	}
	size--;
}
int SeqList::Find(SLDataType x) const
{
	assert(a);
	for (int i = 0; i < size; i++)
	{
		if (a[i] == x) {
			return i;
		}
	}
	return -1;
}
