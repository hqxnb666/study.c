#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
//int Add(int x, int y)
//{
//	return x + y;
//}
//int Sub(int x, int y)
//{
//	return x - y;
//}
//int Mul(int x, int y)
//{
//	return x * y;
//}
//int Div(int x, int y)
//{
//	return x / y;
//}
//void menu()
//{
//	printf("*****1.add*******\n");
//	printf("*****2.sub*******\n");
//	printf("*****3.mul********\n");
//	printf("*****4.div********\n");
//	printf("*****0.exit*******\n");
//		
//}
/*nt  Add(int x, int y)
{
	return x + y;

}*/
 /*void bubble_sort(int arr[], int sz)
{
	int i = 0;
	int temp = 0;

	for (i = 0; i < sz - 1; i++);
	{
		for (int j = 0; j <sz-1-i; j++)
		{
			if (arr[j] > arr[j + 1])
			{
				temp = arr[j];
				arr[j] = arr[j+1];
				arr[j + 1] = temp;

			}
		}
	}
}*/
int main()
{
	////����ָ��-ָ������ָ�� ��ŵ��Ǻ����ĵ�ַ
	////&������ - �����ĵ�ַ
	////������ - �����ĵ�ַ
	//printf("%p", &Add);
 //  int	(*pf)(int, int) = &Add;//pf����һ������ָ�����
 //  int arr[10];
 //  int(*pa)[10] = &arr;//pa��һ������ָ�����


	//int (*pf1)(int, int) = &Add;
	//int (*pf2)(int, int) = &Sub;

	//int (*pfArr[4])(int, int) = { &Add, &Sub }; //��ź���ָ�������
	//�����д��������ͬ�Ķ��Ԫ��
	// 
	// 
	//����ָ������
	
	/*int input;
	int x, y;
	int ret = 0;*/
	//�����Ǳ�׾д��
	/*do
	{
		menu();
		printf("��ѡ��");
		scanf("%d", &input);
		switch (input)
		{
		case 1:
			printf("������2����������");
			scanf("%d %d", &x, &y);
		ret = Add(x, y);
		printf("%d\n", ret);
		
			break;
		case 2:
			printf("������2����������");
			scanf("%d %d", &x, &y);
			ret = Sub(x, y);
			printf("%d\n", ret);

			break;
		case 3:
			printf("������2����������");
			scanf("%d %d", &x, &y);
			ret = Mul(x, y);
			printf("%d\n", ret);

			break;
		case 4:
			printf("������2����������");
			scanf("%d %d", &x, &y);
			ret = Div(x, y);
			printf("%d\n", ret);

			break;
		case 0:
			printf("�Ƴ�������");
			break;
		default:
			printf("ѡ���������ѡ��\n");
			break;

		}
	} while (input);*/

	//����������д��
	
	//do
	//{
	//	menu();
	//	printf("��ѡ��\n");
	//	scanf("%d", &input);
	//	//����ָ������
	//	int(*pfArr[])(int, int) = {  NULL, Add,Sub,Mul,Div };
	//	//                         0     1   2   3   4
	//	if (0 == input)
	//	{
	//		printf("�˳�\n");
	//	}
	//	else if (input >= 1 && input <= 4)
	//	{
	//		printf("������2��������\n");
	//		scanf("%d %d", &x, &y);
	//		ret = pfArr[input](x, y);
	//		printf("%d\n", ret);
	//	}
	//	else
	//	{
	//		printf("ѡ���������ѡ��\n");
	//	}
	//} while (input);


	//ָ����ָ�������ָ��
	//int (*pfArr[5])(int, int) = { 1,2,3,4,5 }; //pfArr�Ǻ���ָ������

	//int (*(*p)[5])(int, int) = &pfArr;




     //�ص�����  һ��ͨ������ָ����õĺ��� 
   

    //qsort -һ���⺯��
//�ײ�ʹ�õĿ�������ķ�ʽ �����ͽ�������
//�����������ֱ��ʹ�ã���������������������������͵�����
   


//ð������
//int arr[] = { 1,2,3,4,5, };
//int sz = sizeof(arr) / sizeof(arr[0]);
//bubble_sort(arr,sz);




void qsort(void* base, //����������ĵ�һ��Ԫ�صĵ�ַ
	size_t num, //�������Ԫ�ظ���
	size_t size,//�����������е�һ��Ԫ�صĴ�С
	int(*compar)(const void*, const void*));//����ָ��  1.��������  2.����ṹ������  1.����ֱ��ʹ��>�Ƚ�
//2.����ֱ��ʹ��

//void*��ָ�벻�ܽ�����Ҳ�����ԼӼ�
//ֻ�ܴ�������������ݵĵ�ַ
int arr[] = { 1,2,3,4,5,6,7,8,9,10 };
qsort(arr,sz��sizeof(arr)[0],)




struct Stu
{
	char name[20];
	int age;
};
int cmp(const void* e1, const void* e2)
{
	((struct Stu*)e1)->age - ((struct Stu*)e2)->age;
}
int cmp_name(const void* e1, const void* e2)
{
	return strcmp(((struct Stu*)e1)->name , ((struct Stu*)e2)->name);
}
void test()
{
	struct Stu arr[] = { {"zhangsan",20},{"lisi",30} };
}
 return 0;
}