#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>

//main - �̶�������
//������ �ǳ�������

//main�����Ǳ����еģ���������ֻ��һ��

//����C���������е��ַ�������Ӣ���ַ�

//int ��������  ˵��Main��������һ������

//int main()
//{
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	return 0;
//}
//
//
//int main()
//{
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	printf("hello word");
//	return 0;
//}



//void main()
//{
//	printf("hh\n");
//}

//int main(void)
//{
//
//}


//int main(int argc, char* argv[])
//{
//	/...
//
//}


//int main() {
//
//	//д����ĵط�
//	printf("hello word\n");
//
//	//printf -- �⺯�� -C���Ա�׼���ṩ��һ���ֳɵĺ���-�ǿ���ֱ��ʹ�õ�
//	//����������Ļ�ϴ�ӡ��Ϣ
//	//�⺯�������Ǳ����������Ӧ��ͷ�ļ��� stdio.h
//
//
//
//	return 0;
//}

//%d -ʮ���Ƶ���ʽ����ӡ����
//int main()
//{
//	printf("%d\n", sizeof(char));
//	printf("%d\n", sizeof(short));
//	printf("%d\n", sizeof(int));
//	printf("%d\n", sizeof(long));
//	printf("%d\n", sizeof(long long));
//	printf("%d\n", sizeof(double));
//	printf("%d\n", sizeof(float));
//
//	return 0;
//}



//ȫ�ֱ����� ��{}��߶���ı�������ȫ�ֱ���
//�ֲ������� ��{}�ڲ�����ı������Ǿֲ�����
//��ǰ�ֲ���ȫ�ֱ�����һ���ط�������ʹ�õ�ͬʱ���ֲ�����
//int a = 10;
//
//int main()
//{
//	int a = 1000;
//	{
//		int b = 0;
//		printf("%d", b);
//	}
//	//printf("%d", a);
//	return 0;
//}


int main()
{
	int a = 0;
	int b = 0;
	int s = 0;
	//����2����
	scanf("%d %d", &a, &b);
	//����
	s = a + b;
	//���
	printf("%d\n", s);

	return 0;
}