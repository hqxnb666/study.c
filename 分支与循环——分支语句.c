#define _CRT_SECURE_NO_WARNINGS
//  c语言是一门 结构化的程序设计语言
// 1. 顺序结构
// 2. 选择结构
// 3. 循环结构
//    分支语句   循环语句   c语言中一个;隔开的就是一条语句
#include <stdio.h>
int main_01()
{
	int age = 10;
	if (age < 18)
	{
		printf("未成年\n");

		printf("不能谈恋爱\n");
	}//要打印多个必须用代码块{}
	else if (age >= 18 && age < 28)
		printf("青年\n");
	else if (age >= 28 && age < 50)
		printf("壮年");
	else
		printf("老不死\n");





	//switch语句
	int day = 0;
	scanf("%d", &day);
	/*if (day == 1)
		printf("星期一\n");
	else if( 2 == day)
		printf("星期2\n");*/




	switch (day)//   必须是整形表达式  case 后必须是整型常量表达式
	{
	case 1:
		printf("星期1\n");
		break;
	case 2:
		printf("星期2\n");
		break;
	case 3:
		printf("星期3\n");
		break;
	case 4:
		printf("星期4\n");//break 跳出循环
		break;
	case 5:
		printf("星期5\n");
		break;
	default:
		printf("输入错误\n");
		break;



		//如果相同 1-5工作日 6-7休息日
		//case后不一定要有
		//  case 1:
		//  case 2:
		//  case 3:
		//  case 4:
		//      printf("工作日\n");
		//   case 6:
		//   case 7:
		//      printf("休息日\n");
	}

	//如果这种太复杂 那我们就换成switch语句
	//   switch(整形表达式)
	//   {
	//       语句项；
	//   }
	// 
	//

	return 0;
}
//  if语法结构
//   if(表达式)
//    语句；

//   if表达式
//   语句1；
//   else
// 语句2；
// 
// 
// 多分支
//   if（表达式1）
//       语句1；
//   else if（表达式2）
//      语句2；
//  else
/   语句3；
