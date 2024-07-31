#define _CRT_SECURE_NO_WARNINGS 1
 static int g_val = 2024;
//全局变量为什么可以跨文件使用
//因为全局变量是具有外部链接属性
//g_val无法使用了
//  static修饰全局变量之后，外部链接属性就会变成内部链接属性
//   其他.c文件无法使用了
static int Add(int x, int y)
 {
	 return x + y;
}
//printf("%d", g_val);