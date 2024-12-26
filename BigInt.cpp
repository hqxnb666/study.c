#include<iostream>
#include<ctime>
#include"BigInt.h"
#include"Rsa.h"

#include"benchmark.h"
#define BENCHMARK_ITERATIONS 5	//5
#define BENCHMARK_ROUNDS 100	//100

using std::cout;
using std::endl;
using std::cin;

void menu()
{//菜单显示函数
	cout << "==========Welcome to use RSA encoder==========" << endl;
	cout << "               e.encrypt 加密              " << endl;
	cout << "               d.decrypt 解密              " << endl;
	cout << "               s.setkey 重置               " << endl;
	cout << "               p.print 显示               " << endl;
	cout << "               q.quit 退出                 " << endl;
	cout << "input your choice:" << endl;
}

bool islegal(const string& str)
{//判断输入是否合法
	for (string::const_iterator it = str.begin(); it != str.end(); ++it)
		if (!isalnum(*it))//不是字母数字
			return false;
	return true;
}

bool decry(Rsa& rsa, string str)
{//解密

	if (!cin || islegal(str) == false)
		return false;
	BigInt c(str);


	BigInt m = rsa.decodeByPr(c);
	/*cout<<"密文:"<<c<<endl
		<<"明文:"<<m<<endl;*/
	return true;
}

bool encry(Rsa& rsa, BigInt& c, string str)
{//加密

	if (!cin || islegal(str) == false)
		return false;
	BigInt m(str);

	c = rsa.encryptByPu(m);

	//cout<<"明文:"<<m<<endl
	//	<<"密文:"<<c<<endl;
	return true;
}

void print(Rsa& rsa)
{
	cout << rsa << endl;
}

void init(Rsa& rsa, int n)
{
	rsa.init(n);
}

int go()
{//控制函数
	char ch;
	string str;
	Rsa rsa;
	BigInt c, m;
	cout << "输入位数:";
	int n;
	cin >> n;


	TIME_BENCH_START("RSA Initialization", BENCHMARK_ITERATIONS);
	TIME_BENCH_ITEM(init(rsa, n / 2), BENCHMARK_ROUNDS);
	TIME_BENCH_FINAL();
	//init(rsa,n/2);
	cout << "初始化完成." << endl;

	while (true)
	{
		menu();//菜单显示
		cout << ">";
		cin >> str;
		if (!cin)
			return 0;

		if (str.length() < 1)
			cout << "重新输入" << endl;
		else
		{
			ch = str.at(0);
			switch (ch)
			{
			case 'e':
			case 'E':
			{
				string str1;
				do
				{
					cout << ">输入16进制数据:";
					cin >> str1;
				} while (cin && str1.length() < 1);

				TIME_BENCH_START("RSA Encryption", BENCHMARK_ITERATIONS);
				TIME_BENCH_ITEM(encry(rsa, c, str1), BENCHMARK_ROUNDS);
				TIME_BENCH_FINAL();
				break;
			}
			case 'd':
			case 'D':
			{
				string str2;
				do
				{
					cout << ">输入16进制数据:";
					cin >> str2;
				} while (cin && str2.length() < 1);

				TIME_BENCH_START("RSA Decryption", BENCHMARK_ITERATIONS);
				TIME_BENCH_ITEM(decry(rsa, str2);, BENCHMARK_ROUNDS);
				TIME_BENCH_FINAL();
				break;

			}

			case 's':
			case 'S':
				go(); // 重新开始初始
				break;
			case 'p':
			case 'P':
				print(rsa); // 输出公私钥信息
				break;
			case 'q':
			case 'Q':
				return 0;
			default:
				cout << "无效的选项，请重新输入" << endl;
				break;
			}

		}
	}
}


int main()
{
	//srand((unsigned int)time(NULL));
	//bench_rsa();
	go();
	//bench_rsa();
}