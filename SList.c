#define _CRT_SECURE_NO_WARNINGS 1
#include "SList.h"
void SLTPrint(SLNode* phead)
{
	SLNode* cur = phead;
	while (cur)		
	{
		printf(" %d-> ", cur->val);
		cur = cur->next;
	}
	printf("NULL\n");
}
SLNode* CreateNode(SLNDataType x)
{
	SLNode* newnode = (SLNode*)malloc(sizeof(SLNode));
	if (newnode == NULL)
	{
		perror("malloc fail");
		exit(-1);
	}
	newnode->val = x;
	newnode->next = NULL;
	return newnode;
}

void SLTPushBack(SLNode** pphead, SLNDataType x)
{
	SLNode* newnode = CreateNode(x);
	if (*pphead == NULL) {
		*pphead = newnode;
	}
	else
	{
		SLNode* tail = *pphead;
		while (tail->next)
		{
			tail = tail->next;
		}
		tail->next = newnode;
	}
}
void SLTPushFront(SLNode** pphead, SLNDataType x)
{
	SLNode* newnode = CreateNode(x);
	newnode->next = *pphead;
	*pphead = newnode;
}
void SLTPopBack(SLNode** pphead)
{
	assert(*pphead);
	if ((*pphead)->next == NULL)
	{
		free(*pphead);
		*pphead == NULL;
	}
	else
	{
		SLNode* tail = *pphead;
		while (tail->next->next)
		{
			tail = tail->next;
		}
		free(tail->next);
		tail->next = NULL;
	}
}
void SLTPopFront(SLNode** pphead)
{
	assert(pphead);
	assert(*pphead);
	SLNode* tmp = *pphead;
	*pphead = (*pphead)->next;
	free(tmp);
}
SLNode* SLTFind(SLNode* phead, SLNDataType x)
{
	SLNode* cur = phead;
	while (cur)
	{
		if (cur->val == x) {
			return cur;
		}
		else
		{
			cur = cur->next;
		}
	}
	return NULL;
}
SLNode* SLTInsert(SLNode** pphead, SLNode* pos, SLNDataType x)
{
	assert(pphead);
	assert(*pphead);
	assert(pos);
	if (*pphead == pos)
	{
		SLTPushFront(pphead,x);
	}
	else
	{
		SLNode* prev = *pphead;
		while (prev->next != pos)
		{
			prev = prev->next;
		}
		SLNode* newnode = CreateNode(x);
		newnode->next = prev->next;
		prev->next = newnode;
	}
}
void SLTErase(SLNode** pphead, SLNode* pos)
{
	assert(pphead);
	assert(*pphead);
	assert(pos);
	if (pos == *pphead)
	{
		SLTPopFront(pphead);
	}
	else
	{
		SLNode* prev = *pphead;
		while (prev->next != pos)
		{
			prev = prev->next;
		}
		prev->next = pos->next;
		free(pos);
	}
}
void SLTDestroy(SLNode** pphead)
{
	assert(pphead);
	SLNode* cur = *pphead;
	while (cur)
	{
		SLNode* next = cur->next;
		free(cur);
		cur = next;
	}
	*pphead = NULL;
}