#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

typedef int LTDataType;

typedef struct ListNode
{
	struct ListNode* prev;
	struct ListNode* next;
	LTDataType val;
}LTNode;

LTNode* LTInit();

void LTPrint(LTNode* phead);
void LTPushBack(LTNode* phead, LTDataType x);
void LTPopBack(LTNode* phead);
void LTPushFront(LTNode* phead, LTDataType x);
void LTPopFront(LTNode* pheadvi);
void LTInsert(LTNode* pos, LTDataType x);
void LTErase(LTNode* pos);
void LTFind(LTDataType x, LTNode* phead);
void LTDestroy(LTNode* phead);
