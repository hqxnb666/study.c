#define _CRT_SECURE_NO_WARNINGS 1>
#include <stdio.h>
int main() {
    int a;
    while (scanf("%d", &a) != EOF) { // 注意 while 处理多个 case
        for (int i = 0; i < a; i++)
        {
            for (int j = 0; j < a; j++)
            {
                if (i == 0 || i == a - 1)
                    printf("* ");
                else if (j == 0 || j == a - 1)
                    printf("* ");
            }
            printf("\n");
        }
    }
    return 0;
}