#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
 //int Add(int x, int y)
//{
	//int z = 0;
	//z = x + y;
	//return z;
//}
int masin()
{
	/*int a = 10;
	int b = 20;
  int sum =	Add(a, b);
  printf("%d\n", sum);*/

  //C�����к����ķ���
  // 1.�⺯��  Io����  �ַ�����������  �ַ���������   �ڴ��������   ʱ��/���ں���  ��ѧ����  �����⺯��
  // 2.�Զ��庯��



    //strcpy �ַ�������
    // char * strcpy (cahr * destination, const cahr * source);
	//char arr1[] = "bit";
	//char arr2[] = "######";
	//strcpy(arr2, arr1);
	//printf("%s\n", arr2);


	//memset  void * memset (void *ptr, int value, size_t num );
	char arr[] = "hello word";
	memset(arr, '*', 5);
	printf("%s\n", arr);
	return 0;
}