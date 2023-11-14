#define _CRT_SECURE_NO_WARNINGS
#include "Ë«Á´±í.h";
LTNode* CreateNode(x)
{
	LTNode* newnode = (LTNode*)malloc(sizeof(LTNode));
	if (newnode == NULL)
	{
		perror("malloc fail:");
	}
	newnode->next = NULL;
	newnode->val = x;
	newnode->prev = NULL;
	return newnode;
}
LTNode* LTInit()
{
	LTNode* phead = CreateNode(-1);
	phead->prev = phead;
	phead->next = phead;
	return phead;

}
void LTPrint(LTNode* phead)
{
	assert(phead);
	
	LTNode* cur = phead->next;
	while (cur != phead)
	{
		printf("%d", cur->val);
		cur = cur->next;
	}

}
void LTPushBack(LTNode* phead, LTDataType x)
{
	assert(phead);
	LTNode* tail = phead->prev;
	LTNode* newhead = CreateNode(x);
	phead->prev = newhead;
	newhead->next = phead;
	tail->next = newhead;
	newhead->prev = tail;
	
}
void LTPopBack(LTNode* phead)
{
	assert(phead);
	assert(phead->next != phead);
	LTNode* tail = phead->prev;
	LTNode* tailprev = tail->prev;
	phead->prev = tailprev;
	tailprev->next = phead;
	free(tail);

}
void LTPushFront(LTNode* phead, LTDataType x)
{
	assert(phead);
	LTNode* newnode = CreateNode(x);
	
	LTNode* next = phead->next;
	phead->next = newnode;
	newnode->prev = phead;
	newnode->next = next;
	next->prev = newnode;
}
void LTPopFront(LTNode* phead)
{
	assert(phead);
	assert(phead->next != phead);
	LTNode* first = phead->next;
	LTNode* second = first->next;
	phead->next = second;
	second->prev = phead;
	free(first);
}
void LTInsert(LTNode* pos, LTDataType x)
{
	assert(pos);
	LTNode* posprev = pos->prev;
	LTNode* newnode = CreateNode(x);
	newnode->next = pos;
	newnode->prev = posprev;
	posprev->next = newnode;
	pos->prev = newnode;
	

}
void LTErase(LTNode* pos)
{
	assert(pos);
	LTNode* posnext = pos->next;
	LTNode* posprev = pos->prev;
	posprev->next = posnext;
	posnext->next = posprev;
	free(pos);
}
void LTFind(LTDataType x, LTNode* phead)
{
	LTNode* cur = phead->next;
	while (cur != phead)
	{
		if (cur->val != x)
			cur = cur->next;
		else
			return cur;
	}
	return NULL;
}
void LTDestroy(LTNode* phead)
{
	assert(phead);
	LTNode* cur = phead->next;
	while (cur->next != phead)
	{
		LTNode* next = cur->next;
		free(cur);
		cur = next;
	}
	free(phead);
}