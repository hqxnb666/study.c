#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>



  void MAX(int a, int b, int c)

{

    if (a > b && b > c)

        printf("%d %d %d", a, b, c);

    else if (a < b && b < c)

        printf("%d %d %d", c, b, a);

    else if (a > c && c > b)

        printf("%d %d %d", a, c, b);

    else if (b > c && c > a)

        printf("%d %d %d", b, c, a);

    else if (b > a && a > c)

        printf("%d %d %d", b, a, c);

    else

        printf("%d %d %d", c, a, b);

}

int ma4in()

{
    int a, b, c;


   

    scanf("%d %d %d", &a, &b, &c );
    MAX(a, b, c);

    return 0;

}