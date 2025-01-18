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
	int size;      // ��Ч����
	int capacity;  // �ռ�����
}SL;

void SLInit(SL* psl);
void SLDestroy(SL* psl);

void SLPrint(SL* psl);
void SLCheckCapacity(SL* psl);

// ͷβ����ɾ��
void SLPushBack(SL* psl, SLDataType x);
void SLPushFront(SL* psl, SLDataType x);
void SLPopBack(SL* psl);
void SLPopFront(SL* psl);

// �����±�λ�õĲ���ɾ��
void SLInsert(SL* psl, int pos, SLDataType x);
void SLErase(SL* psl, int pos);

// �ҵ������±�
// û���ҵ�����-1
int SLFind(SL* psl, SLDataType x);








