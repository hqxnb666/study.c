#define _CRT_SECURE_NO_WARNINGS 1
 static int g_val = 2024;
//ȫ�ֱ���Ϊʲô���Կ��ļ�ʹ��
//��Ϊȫ�ֱ����Ǿ����ⲿ��������
//g_val�޷�ʹ����
//  static����ȫ�ֱ���֮���ⲿ�������Ծͻ����ڲ���������
//   ����.c�ļ��޷�ʹ����
static int Add(int x, int y)
 {
	 return x + y;
}
//printf("%d", g_val);