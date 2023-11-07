#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef int SLNDataType;
typedef struct SListNode
{
	SLNDataType val;
	struct SListNode* next;

}SLNode;
void SLTPrint(SLNode* phead);
void SLTPushBack(SLNode** phead, SLNDataType x);
void SLTPushFront(SLNode** phead, SLNDataType x);
void SLTPopBack(SLNode** phead);
void SLTPopFront(SLNode** phead);
SLNode* SLTFind(SLNode* phead, SLNDataType x);
void SLTInsert(SLNode** phead, SLNode* pos, SLNDataType x);
void SLTErase(SLNode** phead, SLNode* pos);
void SLTInsertAfter(SLNode* pos, SLNDataType x);
void SLTEraseAfter( SLNode* pos);
void SLTDestory(SLNode** pphead);