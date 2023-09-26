#include <stdio.h>

int main()
{
	//三木操作符
	//exp1 ? exp2 : exp3
	//真     计算    不算
	//假      不算    计算
	//逗号表达式  从左到右以此计算 表达最后一个结果


	//  a = get_val();          如果用逗号表达式 改写：
	//                          while(a = get_val(), count_val(a), a>0)
	//                          {
	//count_val(a);             
	//while(a>0)                 }
	/*{
		a = fet_val();
		count_val(a);
	}*/
	


	//下标引用操作符
	/*int arr[] = { 1,2,3,4,5,6 };
	printf("%d", arr[5]);*/

	//成员访问操作符
	//void Print(struct Book* pb)
	//{
	//	printf("%s %d\n", (*pb).name, (*pb).price);
	//	printf("%S %d\n", pb->name, pb->price);
	//	//上下完全相同
	//}

	//表达式求值
	//隐式类型
	//整型提升
	char a = 5;
	char b = 126;
	char c = a + b;
	printf("%d\n", c);
	return 0;
}