#define _CRT_SECURE_NO_WARNINGS
//bug
//debug  找问题的过程叫做“调试” （debug）
// debug被称为调试版本  包含各种调试信息 不进行任何优化
// release 进行各种优化 方便用户进行使用 无需包含调试信息
// 程序员写代码时 需要经常调试代码 建议用debug
// 当程序员写完代码，测试对程序进行测试，知道质量符合交互给用户使用的标准 这时候就要用release
//visual stduio 2022中 在 f5 和 f9搭配使用 直接跳到下一个断点处 f10 和f11的区别是f10通常用来处理一个过程，一个过程可以是函数调用 而f11是每次执行一语句更加细节 可以进入函数内部 可以观察更多细节
// 
// 监视和内存 开始调试后才能看到监视
// 
// 编程常见错误 1.编译错误（发现的是语法问题）2.链接错误 3.运行时错误
//  1编译型错误 编译型错误一般都是语法错误，这类错误一般看错误信息就能我到一些蛛丝马选的，
// 双击错误信息也能初步的跳转到代码错误的地方。编泽错误，随着语言的热练掌握，会越来越少，也容易解决
// 2.链接型错误 看错误提示信息，主要在代码中找到错误信息的标识符 然后定位问题所在
// 一般是因为 1 . 拼写错误 2. 头文件没包含 3. 引用的库不存在
// 3， 运行时错误 运行时错误是千变万化的 需要借助调试，体哦啊是解决的是运行时问题
//
