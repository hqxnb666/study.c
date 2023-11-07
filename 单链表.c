#define _CRT_SECURE_NO_WARNINGS
#include "单链表.h"
void SLTPrint(SLNode* phead)
{
	//不用断言
	SLNode* cur = phead;
	while (cur != NULL)
	{
		printf("%d", cur->val);
		cur = cur->next;
	}
}
SLNode* CreateNode(SLNDataType x)
{
	SLNode* newnode = (SLNode*)malloc(sizeof(SLNode));
	if (newnode == NULL)
	{
		perror("malloc fail:");
		exit(-1);
	}
	newnode->val = x;
	newnode->next = NULL;
	return newnode;
}
void SLTPushBack(SLNode** pphead, SLNDataType x)
{
	assert(pphead);

	SLNode* newnode = CreateNode(x);
		if(*pphead == NULL)
		{
			*pphead = newnode;
		}
		else
		{
			SLNode* tail = *pphead;
			while (tail->next!=NULL)
			{
				tail = tail->next;
			}
			tail->next = newnode;
		}
}
void SLTPushFront(SLNode** pphead, SLNDataType x)
{
	assert(pphead);
	SLNode* newnode = CreateNode(x);
	newnode->next = *pphead;
	*pphead = newnode;
}
void SLTPopBack(SLNode** pphead)
{
	//两种写法  1
	/*assert(*pphead);
	if ((*pphead)->next == NULL)
	{
		free(*pphead);
	}
	else {
		SLNode* prev = NULL;
		SLNode* tail = *pphead;
		while (tail->next != NULL)
		{
			prev = tail;
			tail = tail->next;
		}
		free(tail);
		prev->next = NULL;
	}*/

	assert(pphead);
	assert(*pphead);
	if ((*pphead)->next == NULL)
	{
		free(*pphead);
	}
	else {
		
		SLNode* tail = *pphead;
		while (tail->next->next != NULL)
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
	SLNode* src = *pphead;
	SLNode* des = src->next;
	free(src);
	*pphead = des;
}
SLNode* SLTFind(SLNode* pphead, SLNDataType x)
{
	
	SLNode* cur = pphead;
	while (cur)
	{
		if (cur->val == x)
		{
			return cur;
		}
		else
		{
			cur = cur->next;
		}
	}
	return NULL;
}
void SLTInsert(SLNode** pphead, SLNode* pos, SLNDataType x)
{
	assert( (!pos && !(*pphead)) || pos && *pphead);
	assert(pphead);
	if (*pphead == pos)
	{
		//头插
		SLTPushFront(pphead, x);
	}
	else
	{
		SLNode* prev = *pphead;
		while (prev->next != pos)
		{
			prev = prev->next;
		}
		SLNode* newnode = CreateNode(x);
		prev->next = newnode;
		newnode->next = pos;
	}

}
void SLTErase(SLNode** pphead, SLNode* pos)
{
	
	assert(*pphead);
	assert(pphead);
	assert(pos);
	if (*pphead == pos)
	{
		SLTPopFront(pphead);
	}
	SLNode* prev = *pphead;
	while (prev->next != pos)
	{
		prev = prev->next;
	}
	prev->next = pos->next;
	free(pos);
	pos = NULL;
}
void SLTInsertAfter(SLNode* pos, SLNDataType x)
{
	assert(pos);
	SLNode* newnode = CreateNode(x);
	newnode->next = pos->next;
	pos->next = newnode;
}
void SLTEraseAfter(SLNode* pos)
{
	assert(pos);
	assert(pos->next);
	SLNode* tmp = pos->next;
	pos->next = pos->next->next;
	free(tmp);
	tmp = NULL;
}
void SLTDestory(SLNode** pphead)
{
	assert(pphead);
	SLNode* cur = *pphead;
	while (cur)
	{
		cur = cur->next;
		free(cur);
		cur = NULL;
	}
}