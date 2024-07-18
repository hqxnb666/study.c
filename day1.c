#define _CRT_SECURE_NO_WARNINGS
��дһ���������������ǽ�������ַ�����ת�����������ַ������ַ����� s ����ʽ������

��Ҫ�����������������Ŀռ䣬�����ԭ���޸��������顢ʹ�� O(1) �Ķ���ռ�����һ���⡣



ʾ�� 1��

���룺s = ["h", "e", "l", "l", "o"]
�����["o", "l", "l", "e", "h"]
ʾ�� 2��

���룺s = ["H", "a", "n", "n", "a", "h"]
�����["h", "a", "n", "n", "a", "H"]


��ʾ��

1 <= s.length <= 105
s[i] ���� ASCII ����еĿɴ�ӡ�ַ�
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

���� n �����������ŵĶ������������һ�������������ܹ��������п��ܵĲ��� ��Ч�� ������ϡ�



ʾ�� 1��

���룺n = 3
�����["((()))", "(()())", "(())()", "()(())", "()()()"]
ʾ�� 2��

���룺n = 1
�����["()"]


��ʾ��

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
