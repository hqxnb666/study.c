#define _CRT_SECURE_NO_WARNINGS
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。



示例 1：

输入：s = ["h", "e", "l", "l", "o"]
输出：["o", "l", "l", "e", "h"]
示例 2：

输入：s = ["H", "a", "n", "n", "a", "h"]
输出：["h", "a", "n", "n", "a", "H"]


提示：

1 <= s.length <= 105
s[i] 都是 ASCII 码表中的可打印字符
void reverseString(char* s, int sSize) {
    int begin = 0;
    int end = sSize - 1;

    while (begin <= end)
    {
        int tmp = s[begin];
        s[begin] = s[end];
        s[end] = tmp;
        begin++;
        end--;

    }

}

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。



示例 1：

输入：n = 3
输出：["((()))", "(()())", "(())()", "()(())", "()()()"]
示例 2：

输入：n = 1
输出：["()"]


提示：

1 <= n <= 8
void backtrack(char** ans, int* returnSize, char* cur, int open, int close, int n) {
    if (strlen(cur) == 2 * n) {
        ans[*returnSize] = (char*)malloc(sizeof(char) * (2 * n + 1));
        strcpy(ans[*returnSize], cur);
        (*returnSize)++;
        return;
    }

    if (open < n) {
        cur[strlen(cur)] = '(';
        backtrack(ans, returnSize, cur, open + 1, close, n);
        cur[strlen(cur) - 1] = '\0';
    }
    if (close < open) {
        cur[strlen(cur)] = ')';
        backtrack(ans, returnSize, cur, open, close + 1, n);
        cur[strlen(cur) - 1] = '\0';
    }
}

char** generateParenthesis(int n, int* returnSize) {
    char** ans = (char**)malloc(sizeof(char*) * 1500);
    *returnSize = 0;
    char* cur = (char*)malloc(sizeof(char) * (2 * n + 1));
    memset(cur, '\0', 2 * n + 1);
    backtrack(ans, returnSize, cur, 0, 0, n);
    free(cur);
    return ans;
}
