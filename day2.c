#define _CRT_SECURE_NO_WARNINGS
�������ְ������������ַ� : I�� V�� X�� L��C��D �� M��

�ַ�          ��ֵ
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
���磬 �������� 2 д�� II ����Ϊ�������е� 1 ��12 д�� XII ����Ϊ X + II �� 27 д��  XXVII, ��Ϊ XX + V + II ��

ͨ������£�����������С�������ڴ�����ֵ��ұߡ���Ҳ�������������� 4 ��д�� IIII������ IV������ 1 ������ 5 ����ߣ�����ʾ�������ڴ��� 5 ��С�� 1 �õ�����ֵ 4 ��ͬ���أ����� 9 ��ʾΪ IX���������Ĺ���ֻ�������������������

I ���Է��� V(5) �� X(10) ����ߣ�����ʾ 4 �� 9��
X ���Է��� L(50) �� C(100) ����ߣ�����ʾ 40 �� 90��
C ���Է��� D(500) �� M(1000) ����ߣ�����ʾ 400 �� 900��
����һ���������֣�����ת����������



ʾ�� 1:

����: s = "III"
��� : 3
ʾ�� 2 :

	���� : s = "IV"
	��� : 4
	ʾ�� 3 :

	���� : s = "IX"
	��� : 9
	ʾ�� 4 :

	���� : s = "LVIII"
	��� : 58
	���� : L = 50, V = 5, III = 3.
	ʾ�� 5 :

	���� : s = "MCMXCIV"
	��� : 1994
	���� : M = 1000, CM = 900, XC = 90, IV = 4.
    int romanToInt(char* s) {
    int a[256]; // ���������С�԰��������ַ�
    memset(a, 0, sizeof(a)); // ��ʼ������Ϊ0

    a['I'] = 1;
    a['V'] = 5;
    a['X'] = 10;
    a['L'] = 50;
    a['C'] = 100;
    a['D'] = 500;
    a['M'] = 1000;

    int sum = 0;
    for (int i = 0; i < strlen(s); i++) {
        int value = a[s[i]];
        if (i < strlen(s) - 1 && value < a[s[i + 1]]) {
            sum -= value;
        }
        else {
            sum += value;
        }
    }
    return sum;
}



���������ַ��� s �� t ����дһ���������ж� t �Ƿ��� s ����ĸ��λ�ʡ�

ע�⣺�� s �� t ��ÿ���ַ����ֵĴ�������ͬ����� s �� t ��Ϊ��ĸ��λ�ʡ�



ʾ�� 1:

����: s = "anagram", t = "nagaram"
��� : true
ʾ�� 2 :

    ���� : s = "rat", t = "car"
    ��� : false


    ��ʾ :

    1 <= s.length, t.length <= 5 * 104
    s �� t ������Сд��ĸ


    ���� : ��������ַ������� unicode �ַ���ô�죿���ܷ������Ľⷨ��Ӧ�����������
    int cmp(const void* a, const void* b)
{
    return *(char*)a - *(char*)b;
}
bool isAnagram(char* s, char* t) {
    int lens = strlen(s);
    int lent = strlen(t);
    if (lens != lent)
    {
        return false;
    }
    qsort(s, lens, sizeof(char), cmp);
    qsort(t, lens, sizeof(char), cmp);
    return strcmp(s, t) == 0;
}



����һ���ַ��� columnTitle ����ʾ Excel ����е������ơ����� �������ƶ�Ӧ������� ��

���磺

A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28
...


ʾ�� 1:

����: columnTitle = "A"
��� : 1
ʾ�� 2 :

    ���� : columnTitle = "AB"
    ��� : 28
    ʾ�� 3 :

    ���� : columnTitle = "ZY"
    ��� : 701
    int titleToNumber(char* columnTitle) {
    int number = 0;
    long multiple = 1;
    for (int i = strlen(columnTitle) - 1; i >= 0; i--)
    {
        int k = columnTitle[i] - 'A' + 1;
        number += k * multiple;
        multiple *= 26;
    }
    return number;
}�������ְ������������ַ��� I�� V�� X�� L��C��D �� M��

�ַ�          ��ֵ
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
���磬 �������� 2 д�� II ����Ϊ�������е� 1��12 д�� XII ����Ϊ X + II �� 27 д��  XXVII, ��Ϊ XX + V + II ��

ͨ������£�����������С�������ڴ�����ֵ��ұߡ���Ҳ�������������� 4 ��д�� IIII������ IV������ 1 ������ 5 ����ߣ�����ʾ�������ڴ��� 5 ��С�� 1 �õ�����ֵ 4 ��ͬ���أ����� 9 ��ʾΪ IX���������Ĺ���ֻ�������������������

I ���Է��� V(5) �� X(10) ����ߣ�����ʾ 4 �� 9��
X ���Է��� L(50) �� C(100) ����ߣ�����ʾ 40 �� 90��
C ���Է��� D(500) �� M(1000) ����ߣ�����ʾ 400 �� 900��
����һ������������תΪ�������֡�



ʾ�� 1:

����: num = 3
��� : "III"
ʾ�� 2 :

    ���� : num = 4
    ��� : "IV"
    ʾ�� 3 :


    const int values[] = { 1000,900,500,400,100,90,50,40,10,9,5,4,1 };
const char* symbols[] = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };

char* intToRoman(int num) {
    char* roman = malloc(sizeof(char) * 16);
    roman[0] = '\0';
    for (int i = 0; i < 13; i++)
    {
        while (num >= values[i])
        {
            num -= values[i];
            strcpy(roman + strlen(roman), symbols[i]);
        }
        if (num == 0)
            break;
    }
    return roman;
}