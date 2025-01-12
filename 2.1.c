#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>

//
//   2.1
// 
//int main()
//{
//    int height = 6;
//    for (int i = 0; i < height; i++)
//    {
//        for (int j = 0; j < 8; j++)
//        {
//            if (i < 3 && j == 8 - i-1) {
//                printf("*");
//            }
//            else if (i >= 3 && j == i - 3)
//            {
//                printf("*");
//            }
//            else if (i >= 3 && j == 8-i-1)
//            {
//                printf("*");
//            }
//            else {
//                printf(" ");
//            }
//        }
//        printf("\n");
//    }
//    return 0;
//}



//  2.2 - 2.3

//int main()
//{
//	float r = 0;
//	scanf("%f", &r);
//	float v = 4.0f / 3.0f * 3.14f * r * r * r;
//	printf("球体体积v:%.2f\n", v);
//	return 0;
//}


//2.4

//int main()
//{
//	float money = 0.0;
//	printf("Enter an amount: ");
//	scanf("%f", &money);
//	printf("With tax added: $%.2f\n", money * 1.05f);
//	return 0;
//}

//2.5

//int main()
//{
//	int a = 0;
//	scanf("%d", &a);
//	a = 3 * a * a * a * a * a + 2 * a * a * a * a - 5 * a * a * a - a * a + 7 * a - 6;
//	printf("result is : %d", a);
//	return 0;
//}

//2.6

//int main()
//{
//	int x = 0;
//	scanf("%d", &x);
//	x = ((((3 * x + 2) * x - 5) * x - 1) * x + 7) * x - 6;
//	printf("%d", x);
//	return 0;
//}

//2.7

//int main()
//{
//	int a = 0;
//	int b20, b10, b5, b1;
//	printf("Enter a dollar amout: ");
//	scanf("%d", &a);
//	b20 = a / 20;
//	a %= 20;
//	b10 = a / 10;
//	a %= 10;
//	b5 = a / 5;
//	a %= 5;
//	b1 = a / 1;
//	printf("$20 bills : %d\n", b20);
//	printf("$10 bills : %d\n", b10);
//	printf("$5  bills : %d\n", b5);
//	printf("$1  bills : %d\n", b1);
//
//	return 0;
//}


//2.8

#include <stdio.h>

int main() {
    float loanAmount, interestRate, monthlyPayment;
    printf("Enter amount of loan: \n");
    scanf("%f", &loanAmount);
    printf("Enter interest rate: \n");
    scanf("%f", &interestRate);
    printf("Enter monthly payment: \n");
    scanf("%f", &monthlyPayment);
 
    // 第一个月的余额计算
    float monthlyInterestRate = (interestRate / 100.0) / 12;  // 每月利率
    float first = loanAmount - monthlyPayment + (loanAmount * monthlyInterestRate);
    first = first < 0 ? 0 : first;  // 避免余额小于0的情况

    // 第二个月的余额计算
    float second = first - monthlyPayment + (first * monthlyInterestRate);
    second = second < 0 ? 0 : second;  // 避免余额小于0的情况

    // 第三个月的余额计算
    float third = second - monthlyPayment + (second * monthlyInterestRate);
    third = third < 0 ? 0 : third;  // 避免余额小于0的情况

    printf("Balance remaining after first payment: $%.2f\n", first);
    printf("Balance remaining after second payment: $%.2f\n", second);
    printf("Balance remaining after third payment: $%.2f\n", third);

    return 0;
}
