#pragma once

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>


typedef int SLDataType;
//typedef int shujuleixing;

// sequence list
typedef struct SeqList
{
	SLDataType* a;
	int size;      // 有效数据
	int capacity;  // 空间容量
}SL;

void SLInit(SL* psl);
void SLDestroy(SL* psl);

void SLPrint(SL* psl);
void SLCheckCapacity(SL* psl);

// 头尾插入删除
void SLPushBack(SL* psl, SLDataType x);
void SLPushFront(SL* psl, SLDataType x);
void SLPopBack(SL* psl);
void SLPopFront(SL* psl);

// 任意下标位置的插入删除
void SLInsert(SL* psl, int pos, SLDataType x);
void SLErase(SL* psl, int pos);

// 找到返回下标
// 没有找到返回-1
int SLFind(SL* psl, SLDataType x);








