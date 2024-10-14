#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
void PrintMulTable(int N)
{
    for (int i = 1; i <= N; ++i)
    {
        for (int j = 1; j <= i; ++j)
        {
            printf("%d*%d=%2d  ", j, i, j * i);
        }
        printf("\n");
    }
}
int main() {
    int a;
    scanf("%d", &a);
    PrintMulTable(a);
    return 0;
}