#define _CRT_SECURE_NO_WARNINGS
#include "Date.h"



Date::Date(int year , int month , int day)
{
	_year = year;
	_month = month;
	_day = day;

	if (!CheckInvalid())
	{
		cout << "构造的日期非法" << endl;
	}
}
bool Date::operator<(const Date& d)
{
	if (_year < d._year) {
		return true;
	}
	else if (_year == d._year) {
		if (_month < d._month) {
			return true;
		}
		else if (_month == d._month) {
			if (_day < d._day) {
				return true;
			}
		}
	}
	return false;
}

bool  Date::operator<=(const Date& d)
{
	return *this < d || *this == d;
}
bool Date::operator>(const Date& d)
{
	return !(*this <= d);
}
bool Date::operator>=(const Date& d)
{
	return !(*this < d);
}
bool Date::operator==(const Date& d)
{
	return _year == d._year && _month == d._month && _day == d._day;
}
bool Date::operator!=(const Date& d)
{
	return !(*this == d);
}

//Date& Date::operator += ( int day)
//{
//	_day += day;
//	while (_day > GetMonthDay(_year, _month))
//	{
//		_day -= GetMonthDay(_year, _month);
//		++_month;
//		if (_month == 13)
//		{
//			++_year;
//			_month = 1;
//		}
//	}
//	return *this;
//}
Date Date::operator + (int day)
{
	Date tmp(*this);
	tmp._day += day;
	while (tmp._day > GetMonthDay(_year, _month))
	{
		tmp._day -= GetMonthDay(_year, _month);
		++tmp._month;
		if (tmp._month == 13)
		{
			++tmp._year;
			tmp._month = 1;
		}
	}
	return tmp;
}
Date& Date::operator += (int day)
{
	*this = *this + day;   //这种没有第一张+=好，，因为他在调用+的时候要产生临时变量
	return *this;
}

Date Date:: operator-(int day)
{
	Date tmp = *this;
	tmp -= day;
	return tmp;
}
Date& Date:: operator-=(int day)
{
	_day -= day;
	while (_day <= 0)
	{
		--_month;
		if (_month == 0)
		{
			--_year;
			_month = 12;
		}
		_day += GetMonthDay(_year, _month);
	}
	return *this;
}

Date& Date::operator++()
{
	*this += 1;
	return *this;
}

Date Date::operator++(int)
{
	Date tmp = *this;
	*this += 1;
	return tmp;
}

Date& Date::operator--()
{
	*this -= 1;
	return *this;
}
Date Date::operator--(int)
{
	Date tmp = *this;
	*this -= 1;
	return tmp;
}


Date& Date::operator=(const Date& d)
{
	_year = d._year;
	_month = d._month;
	_day = d._day;
	return *this;
}
int Date::operator-(const Date& d)//日期-日期
{
	int flag = 1;
	int n = 1;
	Date max = d;
	Date min = *this;
	if (max < min)
	{
		flag = -1;
		max = *this;
		min = d;
	}
	while (min != max)
	{
		++min;
		++n;
	}
	return n * flag;
}
bool Date::CheckInvalid()
{
	if (_year <= 0 || _month <1 || _month > 12 || _day <1 || _day > GetMonthDay(_year, _month))
	{
		return false;
	}
	else
	{
		return true;
	}
}

ostream& operator<<(ostream& out, const Date& d)
{
	out << d._year << "年" << d._month << "月" << d._day << "日" << endl;
	return out;
}
istream& operator>>(istream& in, Date& d)

{
	while (true)
	{
		in >> d._year >> d._month >> d._day;
		if (!d.CheckInvalid())
		{
			cout << "输入无效，请重新输入" << endl;
		}
		else
		{
			break;
		}
	}
	return in;
	
}