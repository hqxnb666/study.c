#include <stdio.h>

int main()
{
	//��ľ������
	//exp1 ? exp2 : exp3
	//��     ����    ����
	//��      ����    ����
	//���ű��ʽ  �������Դ˼��� ������һ�����


	//  a = get_val();          ����ö��ű��ʽ ��д��
	//                          while(a = get_val(), count_val(a), a>0)
	//                          {
	//count_val(a);             
	//while(a>0)                 }
	/*{
		a = fet_val();
		count_val(a);
	}*/
	


	//�±����ò�����
	/*int arr[] = { 1,2,3,4,5,6 };
	printf("%d", arr[5]);*/

	//��Ա���ʲ�����
	//void Print(struct Book* pb)
	//{
	//	printf("%s %d\n", (*pb).name, (*pb).price);
	//	printf("%S %d\n", pb->name, pb->price);
	//	//������ȫ��ͬ
	//}

	//���ʽ��ֵ
	//��ʽ����
	//��������
	char a = 5;
	char b = 126;
	char c = a + b;
	printf("%d\n", c);
	return 0;
}