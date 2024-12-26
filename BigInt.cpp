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
{//�˵���ʾ����
	cout << "==========Welcome to use RSA encoder==========" << endl;
	cout << "               e.encrypt ����              " << endl;
	cout << "               d.decrypt ����              " << endl;
	cout << "               s.setkey ����               " << endl;
	cout << "               p.print ��ʾ               " << endl;
	cout << "               q.quit �˳�                 " << endl;
	cout << "input your choice:" << endl;
}

bool islegal(const string& str)
{//�ж������Ƿ�Ϸ�
	for (string::const_iterator it = str.begin(); it != str.end(); ++it)
		if (!isalnum(*it))//������ĸ����
			return false;
	return true;
}

bool decry(Rsa& rsa, string str)
{//����

	if (!cin || islegal(str) == false)
		return false;
	BigInt c(str);


	BigInt m = rsa.decodeByPr(c);
	/*cout<<"����:"<<c<<endl
		<<"����:"<<m<<endl;*/
	return true;
}

bool encry(Rsa& rsa, BigInt& c, string str)
{//����

	if (!cin || islegal(str) == false)
		return false;
	BigInt m(str);

	c = rsa.encryptByPu(m);

	//cout<<"����:"<<m<<endl
	//	<<"����:"<<c<<endl;
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
{//���ƺ���
	char ch;
	string str;
	Rsa rsa;
	BigInt c, m;
	cout << "����λ��:";
	int n;
	cin >> n;


	TIME_BENCH_START("RSA Initialization", BENCHMARK_ITERATIONS);
	TIME_BENCH_ITEM(init(rsa, n / 2), BENCHMARK_ROUNDS);
	TIME_BENCH_FINAL();
	//init(rsa,n/2);
	cout << "��ʼ�����." << endl;

	while (true)
	{
		menu();//�˵���ʾ
		cout << ">";
		cin >> str;
		if (!cin)
			return 0;

		if (str.length() < 1)
			cout << "��������" << endl;
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
					cout << ">����16��������:";
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
					cout << ">����16��������:";
					cin >> str2;
				} while (cin && str2.length() < 1);

				TIME_BENCH_START("RSA Decryption", BENCHMARK_ITERATIONS);
				TIME_BENCH_ITEM(decry(rsa, str2);, BENCHMARK_ROUNDS);
				TIME_BENCH_FINAL();
				break;

			}

			case 's':
			case 'S':
				go(); // ���¿�ʼ��ʼ
				break;
			case 'p':
			case 'P':
				print(rsa); // �����˽Կ��Ϣ
				break;
			case 'q':
			case 'Q':
				return 0;
			default:
				cout << "��Ч��ѡ�����������" << endl;
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