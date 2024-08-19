#include<stdio.h>
#include <stdlib.h>
char* mapping[] = { "","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz" };
void backtrack(char* digits, int index, char* current, int currentSize, char** result, int* returnSize, int digitsLength) {
    if (currentSize == digitsLength) {
        result[*returnSize] = (char*)malloc(sizeof(char) * (digitsLength + 1));
        strcpy(result[*returnSize], current);
        (*returnSize)++;
        return;
    }
    char* letters = mapping[digits[index] - '0'];
    for (int i = 0; letters[i] != '\0'; i++) {
        current[currentSize] = letters[i];
        backtrack(digits, index + 1, current, currentSize + 1, result, returnSize, digitsLength);
        // 不需要在这里减小currentSize，因为下次循环或递归调用会覆盖它
    }
}
char** letterCombinations(char* digits, int* returnSize) {
    *returnSize = 0;
    int digitsLength = strlen(digits);
    if (digitsLength == 0) {
        return NULL;
    }
    char** result = (char**)malloc(sizeof(char*) * 10000);
    char* current = (char*)malloc((digitsLength + 1) * sizeof(char));
    backtrack(digits, 0, current, 0, result, returnSize, digitsLength);

    free(current);  // 移动到这里
    return result;
}
void printResult(char** result, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%s", result[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    // 测试用例 1: 普通情况
    char* digits1 = "23";
    int returnSize1;
    char** result1 = letterCombinations(digits1, &returnSize1);
    printf("Test Case 1: \"23\" -> ");
    printResult(result1, returnSize1);
    // 释放内存
    for (int i = 0; i < returnSize1; i++) free(result1[i]);
    free(result1);

    // 测试用例 2: 单个数字
    char* digits2 = "2";
    int returnSize2;
    char** result2 = letterCombinations(digits2, &returnSize2);
    printf("Test Case 2: \"2\" -> ");
    printResult(result2, returnSize2);
    // 释放内存
    for (int i = 0; i < returnSize2; i++) free(result2[i]);
    free(result2);

    // 测试用例 3: 空字符串
    char* digits3 = "";
    int returnSize3;
    char** result3 = letterCombinations(digits3, &returnSize3);
    printf("Test Case 3: \"\" -> ");
    printResult(result3, returnSize3);
    // 释放内存
    for (int i = 0; i < returnSize3; i++) free(result3[i]);
    free(result3);

    // 测试用例 4: 包含4个数字
    char* digits4 = "2796";
    int returnSize4;
    char** result4 = letterCombinations(digits4, &returnSize4);
    printf("Test Case 4: \"2796\" -> ");
    printResult(result4, returnSize4);
    // 释放内存
    for (int i = 0; i < returnSize4; i++) free(result4[i]);
    free(result4);

    return 0;
}