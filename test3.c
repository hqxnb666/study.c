#define  _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>

void Swap(int* x1, int* x2)
{
	int tmp = *x1;
	*x1 = *x2;
	*x2 = tmp;
}

void AdjustDown(int* a, int size, int parent)
{
	int child = parent * 2 + 1;
	while (child < size)
	{
		if (child + 1 < size && a[child] > a[child + 1])
		{
			child++;
		}
		if (a[parent] > a[child])
		{
			Swap(&a[parent], &a[child]);
			parent = child;
			child = child * 2 + 1;
		}
		else
		{
			break;
		}
	}
}

//int main()
//{
//	int a[] = { 23,26,21,4,2,5,6,1,3,8,7,12,19,51,16,18,21 };
//	int sz = sizeof(a) / sizeof(a[0]);
//
//	//建大堆
//	int i = 0;
//	for (i = (sz - 1 - 1) / 2; i >= 0; i--)
//	{
//		AdjustDown(a, sz, i);
//	}
//	//
//	int end = sz - 1;
//	while (end>0)
//	{
//		Swap(&a[0], &a[end]);
//		AdjustDown(a, end, 0);
//		end--;
//	}
//
//	for (i = 0; i < sz; i++)
//	{
//		printf("%d ",a[i]);
//	}
//	printf("\n");
//
//
//	return 0;
//}

void AdjustUp(int* a, int child)
{
	int parent = (child - 1) / 2;
	while (child > 0)
	{
		if (a[parent] > a[child])
		{
			Swap(&a[parent], &a[child]);
			child = parent;
			parent = (child - 1) / 2;
		}
		else
		{
			break;
		}
	}

}

void PrintTopK(const char* pf, int k)
{
	FILE* f = fopen(pf, "r");
	if (f == NULL)
	{
		perror("fopen fail");
		return;
	}

	int* minheap = (int*)malloc(sizeof(int) * k);
	if (minheap == NULL)
	{
		perror("malloc fail");
		return;
	}
	//小堆
	int i = 0;
	for (i = 0; i < k; i++)
	{
		fscanf(f, "%d", &minheap[i]);
		AdjustUp(minheap, i);
	}
	for (i = 0; i < k; i++) {
		if (fscanf(f, "%d", &minheap[i]) == 1) {
			AdjustUp(minheap, i);
		}
		else {
			// 处理错误，例如通过打印错误消息、关闭文件和释放资源
			perror("fscanf failed to read an integer");
			free(minheap);
			fclose(f);
			return;
		}
	}

	int x = 0;


	int result;
	while ((result = fscanf(f, "%d", &x)) != EOF) {
		if (result == 1 && x > minheap[0]) {
			minheap[0] = x;
			AdjustDown(minheap, k, 0);
		}
		else if (result != 1) {
			// 处理错误情况
			perror("fcanf");
			break;
		}
	}


	for (i = 0; i < k; i++)
	{
		printf("%d ", minheap[i]);
	}
	printf("\n");

	free(minheap);
	fclose(f);

}

int main()
{
	PrintTopK("data.txt", 8);
	return 0;
}