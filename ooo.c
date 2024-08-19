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
        // ����Ҫ�������СcurrentSize����Ϊ�´�ѭ����ݹ���ûḲ����
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

    free(current);  // �ƶ�������
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
    // �������� 1: ��ͨ���
    char* digits1 = "23";
    int returnSize1;
    char** result1 = letterCombinations(digits1, &returnSize1);
    printf("Test Case 1: \"23\" -> ");
    printResult(result1, returnSize1);
    // �ͷ��ڴ�
    for (int i = 0; i < returnSize1; i++) free(result1[i]);
    free(result1);

    // �������� 2: ��������
    char* digits2 = "2";
    int returnSize2;
    char** result2 = letterCombinations(digits2, &returnSize2);
    printf("Test Case 2: \"2\" -> ");
    printResult(result2, returnSize2);
    // �ͷ��ڴ�
    for (int i = 0; i < returnSize2; i++) free(result2[i]);
    free(result2);

    // �������� 3: ���ַ���
    char* digits3 = "";
    int returnSize3;
    char** result3 = letterCombinations(digits3, &returnSize3);
    printf("Test Case 3: \"\" -> ");
    printResult(result3, returnSize3);
    // �ͷ��ڴ�
    for (int i = 0; i < returnSize3; i++) free(result3[i]);
    free(result3);

    // �������� 4: ����4������
    char* digits4 = "2796";
    int returnSize4;
    char** result4 = letterCombinations(digits4, &returnSize4);
    printf("Test Case 4: \"2796\" -> ");
    printResult(result4, returnSize4);
    // �ͷ��ڴ�
    for (int i = 0; i < returnSize4; i++) free(result4[i]);
    free(result4);

    return 0;
}