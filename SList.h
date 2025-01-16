#pragma once
#include<stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef int SLNDataType;
typedef struct SListNode
{
	SLNDataType val;
	struct SListNode* next;
}SLNode;

void SLTPrint(SLNode* phead);
void SLTPushBack(SLNode** pphead, SLNDataType x);

void SLTPopBack(SLNode** pphead);
void SLTPopFront(SLNode** pphead);

SLNode* SLTFind(SLNode* phead, SLNDataType x);
SLNode* SLTInsert(SLNode** pphead, SLNode* pos, SLNDataType x);

void SLTErase(SLNode** pphead, SLNode* pos);
void SLTDestroy(SLNode** pphead);
