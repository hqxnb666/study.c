//#define _CRT_SECURE_NO_WARNINGS 1
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//void swap(void* buf1, void* buf2, size_t size)
//{
//	int i = 0;
//	for (i = 0; i < size; i++)
//	{
//		char tmp = *((char*)buf1 + i);
//		*((char*)buf1 + i) = *((char*)buf2 + i);
//		*((char*)buf2 + i) = tmp;
//	}
//}
//void pprint(struct Stu arr[], int sz)
//{
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%s", arr[i].name);
//	}
//	printf("\n");
//}
//void bubble_sort(void* base, size_t num,size_t size,int(*cmp)(const void*e1, const void* e2))
//{
//	//ð����������
//	int i = 0;
//	for (i = 0; i <num-1; i++)
//	{
//		int j = 0;
//		for (j = 0; j < num - i - 1; j++)
//		{
//			if (cmp((char*)base + j * size, (char*)base + (j + 1) * size)> 0)
//			{
//				//����
//				swap((char*)base + j * size, (char*)base + (j + 1) * size, size);
//			}
//		}
//	}
//}
//
//struct Stu
//{
//	char name[20];
//	int age;
//};
//
//int cmp(const void* e1, const void* e2)
//{
//	return *(int*)e1 - *(int*)e2;
//}
//void test1()
//{
//	int arr[] = { 6,7,8,9,10,2,3,4,5 };
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	bubble_sort(arr, sz, sizeof(arr[0]), cmp);
//	pprint(arr, sz);
//}
//int cmp_age(const void* e1, const void* e2)
//{
//	return ((struct Stu*)e1)->age - ((struct Stu*)e2)->age;
//}
//int cmp_name(const void* e1, const void* e2)
//{
//	return strcmp(((struct Stu*)e1)->name , ((struct Stu*)e2)->name);
//}
//void test2()
//{
//	struct Stu arr[] = { {"zhangsan",20}, {"lisi",30},{"wangwu",60} };
//	int sz = sizeof(arr) / sizeof(arr[0]);
//	bubble_sort(arr, sz, sizeof(arr[0]), cmp_name);
//	pprint(arr, sz);
//}
//int main()
//{
//	//ð������ ʵ��һ��������
//	//test1();//��������
//	test2();//���Խṹ
//	
//	return 0;
//}





//#define _CRT_SECURE_NO_WARNINGS 1
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//
//
//// ���� Stu �ṹ��
//struct Stu
//{
//    char name[20];
//    int age;
//};
//
//// ��������
//void swap(void* buf1, void* buf2, size_t size)
//{
//    int i;
//    for (i = 0; i < size; i++)
//    {
//        char tmp = *((char*)buf1 + i);
//        *((char*)buf1 + i) = *((char*)buf2 + i);
//        *((char*)buf2 + i) = tmp;
//    }
//}
//
//// ��ӡ�ṹ������
//void pprint(struct Stu arr[], int sz)
//{
//    int i;
//    for (i = 0; i < sz; i++)
//    {
//        printf("%s\n", arr[i].name);
//    }
//}
//
//// ð������
//void bubble_sort(void* base, size_t num, size_t size, int(*cmp)(const void* e1, const void* e2))
//{
//    int i, j;
//    for (i = 0; i < num - 1; i++)
//    {
//        for (j = 0; j < num - i - 1; j++)
//        {
//            if (cmp((char*)base + j * size, (char*)base + (j + 1) * size) > 0)
//            {
//                // ����
//                swap((char*)base + j * size, (char*)base + (j + 1) * size, size);
//            }
//        }
//    }
//}
//
//// �ȽϺ�������������
//int cmp_age(const void* e1, const void* e2)
//{
//    return ((struct Stu*)e1)->age - ((struct Stu*)e2)->age;
//}
//
//// �ȽϺ�������������
//int cmp_name(const void* e1, const void* e2)
//{
//    return strcmp(((struct Stu*)e1)->name, ((struct Stu*)e2)->name);
//}
//
//void test1()
//{
//    int arr[] = { 6, 7, 8, 9, 10, 2, 3, 4, 5 };
//    int sz = sizeof(arr) / sizeof(arr[0]);
//    bubble_sort(arr, sz, sizeof(arr[0]), cmp_age);
//    // ʹ����ȷ�ĺ�����ӡ��������
//    for (int i = 0; i < sz; i++) {
//        printf("%d ", arr[i]);
//    }
//    printf("\n");
//}
//
//void test2()
//{
//    struct Stu arr[] = { {"zhangsan", 20}, {"lisi", 30}, {"wangwu", 60} };
//    int sz = sizeof(arr) / sizeof(arr[0]);
//    bubble_sort(arr, sz, sizeof(arr[0]), cmp_name);
//    pprint(arr, sz);
//}
//
//int main()
//{
//    // ���ò��Ժ���
//    test1(); // ��������
//    test2(); // ���Խṹ��
//
//    return 0;
//}





//#include <stdio.h>
//int main()
//{
//	int a[] = { 1,2,3,4 };
//
//	printf("%d\n", sizeof(a));//16  ������a��������sizeof�ڲ�����������ʾ�������飬��������������鵥λ���ֽ���16�ֽ�
//
//	printf("%d\n", sizeof(a + 0));//a���ǵ��Ķ�����sizeof�ڲ� Ҳû��& ��������������Ԫ�ص�ַ a+0 ������Ԫ�ص�ַ
//	//�ǵ�ַ����4/8 byte
//	printf("%d\n", sizeof(*a)); //a����Ԫ�ص�ַ *a������Ԫ��  ����4Byte  
//	//    *a == *(a+0) == a[0]
//	printf("%d\n", sizeof(a+1));//a���ǵ��Ķ�����sizeof�ڲ� Ҳû��& ��������������Ԫ�ص�ַ a+1���ǵڶ���Ԫ�صĵ�ַ
//	//a+1 == &a[1]  �ǵڶ���Ԫ�صĵ�ַ �ǵ�ַ����4/8���ֽ�
//
//	printf("%d\n", sizeof(a[1]));//4
//	printf("%d\n", sizeof(&a)); //��ȡ����ĵ�ַ ��������ĵ�ַҲ�ǵ�ַ ��ַ����4/8
//	//����ĵ�ַ �� ������Ԫ�صĵ�ַ ������������͵�����
//	//a  --  int*    int * p = a
//	//&a  -- int (*)[4]    int (*p)[4] = &a
//	printf("%d\n", sizeof(*&a)); //16  ȡ��������& *�໥����
//	printf("%d\n", sizeof(&a + 1));//4/8  ���ǵ�ַ 
//	printf("%d\n", sizeof(&a[0]));//����Ԫ�ص�ַ ������ǵ�ַ�Ĵ�С 4/8
//	printf("%d\n", sizeof(&a[0] + 1));//4/8
//	return 0;
//}


//#include <stdio.h>
//
//int main()
//{
//	char arr[] = { 'a', 'b', 'c', 'd', 'e', 'f' };
//
//	printf("%d\n", sizeof(arr));// 6   ������ar��������sizeof�ڲ����������������Ĵ�С 
//
//	printf("%d\n", sizeof(arr + 0));// 4/8   arr����Ԫ�ص�ַ == &arr[0]
//	printf("%d\n", sizeof(*arr));//   arr����Ԫ�ص�ַ *arr������Ԫ�� 1
//	printf("%d\n", sizeof(arr[1]));//;1
//	printf("%d\n", sizeof(&arr));//4/8
//	printf("%d\n", sizeof(&arr + 1));//4/8
//	printf("%d\n", sizeof(&arr[0] + 1));//4/8
//
//	//strlen���ַ�������
//	//ͳ�Ƶ���\0����֮ǰ���ַ�������
//	printf("%d\n", strlen(arr));//���ֵ  arr����Ԫ�ص�ַ
//	printf("%d\n", strlen (arr));//
//	printf("%d\n", strlen(arr + 0));//arr+0������Ԫ�ص�ַ
//	printf("%d\n", strlen(*arr));//��Ԫ��  == ��a��-97  
//	//վ��strlen�Ƕ� ���ν�ȥ��'a'-97���ǵ�ַ ��97��Ϊ��ַ ֱ�ӽ��з��ʾ��ǷǷ�����
//	printf("%d\n", strlen(arr[1]));//��b��-98 ͬ���Ǹ�����Ĵ���
//	printf("%d\n", strlen(&arr));//&arr -- char(*)[6]
//	//const char*   �Ǹ����ֵ
//	printf("%d\n", strlen(& arr + 1));//Ҳ�Ǹ����ֵ
//	printf("%d\n", strlen(&arr[0] + 1));//Ҳ�Ǹ����ֵ
//
//	return 0;
//}



#include <stdio.h>
int main()
{

	return 0;
}