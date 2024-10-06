#define _CRT_SECURE_NO_WARNINGS
罗马数字包含以下七种字符 : I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V(5) 和 X(10) 的左边，来表示 4 和 9。
X 可以放在 L(50) 和 C(100) 的左边，来表示 40 和 90。
C 可以放在 D(500) 和 M(1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。



示例 1:

输入: s = "III"
输出 : 3
示例 2 :

	输入 : s = "IV"
	输出 : 4
	示例 3 :

	输入 : s = "IX"
	输出 : 9
	示例 4 :

	输入 : s = "LVIII"
	输出 : 58
	解释 : L = 50, V = 5, III = 3.
	示例 5 :

	输入 : s = "MCMXCIV"
	输出 : 1994
	解释 : M = 1000, CM = 900, XC = 90, IV = 4.
    int romanToInt(char* s) {
    int a[256]; // 增加数组大小以包含所有字符
    memset(a, 0, sizeof(a)); // 初始化数组为0

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



给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。



示例 1:

输入: s = "anagram", t = "nagaram"
输出 : true
示例 2 :

    输入 : s = "rat", t = "car"
    输出 : false


    提示 :

    1 <= s.length, t.length <= 5 * 104
    s 和 t 仅包含小写字母


    进阶 : 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？
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



给你一个字符串 columnTitle ，表示 Excel 表格中的列名称。返回 该列名称对应的列序号 。

例如：

A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28
...


示例 1:

输入: columnTitle = "A"
输出 : 1
示例 2 :

    输入 : columnTitle = "AB"
    输出 : 28
    示例 3 :

    输入 : columnTitle = "ZY"
    输出 : 701
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
}罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V(5) 和 X(10) 的左边，来表示 4 和 9。
X 可以放在 L(50) 和 C(100) 的左边，来表示 40 和 90。
C 可以放在 D(500) 和 M(1000) 的左边，来表示 400 和 900。
给你一个整数，将其转为罗马数字。



示例 1:

输入: num = 3
输出 : "III"
示例 2 :

    输入 : num = 4
    输出 : "IV"
    示例 3 :


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