#define _CRT_SECURE_NO_WARNINGS 1
//#include <stdio.h>
//int main()
//{
//	printf("%6d,%4d\n",86,1040);
//	printf("%12.5e\n", 30.253);
//	printf("%.4f\n",83.162);
//	printf("%-6.2g\n", .0000009979);
//	return 0;
//}
#include <stdio.h>

int main(void) {
    int beg, mid, end;

    printf("Enter phone number [(xxx) xxx-xxxx]: ");
    scanf(" (%d ) %d- %d", &beg, &mid, &end);
    printf("You entered %.3d.%3d.%.4d\n", beg, mid, end);

    return 0;
}
