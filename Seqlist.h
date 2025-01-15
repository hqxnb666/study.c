#pragma once
#include <iostream>
#include <cstdlib>
typedef int SLDataType;

class SeqList
{
public:
    SeqList();
    ~SeqList();

    void Print() const;
    void PushBack(SLDataType x);
    void PushFront(SLDataType x);
    void PopBack();
    void PopFront();
    void Insert(int pos, SLDataType x);
    void Erase(int pos);
    int Find(SLDataType x) const; // 找到返回索引，否则-1

private:
    void CheckCapacity();
    SLDataType* a;
    int size;
    int capacity;
};
