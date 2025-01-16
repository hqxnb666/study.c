#include "SList.h"

int main() {
    SLNode* head = NULL;

    // 测试尾插
    SLTPushBack(&head, 1);
    SLTPushBack(&head, 2);
    SLTPushBack(&head, 3);
    SLTPrint(head); // 输出: 1 -> 2 -> 3 -> NULL

    // 测试头插
    SLTPushFront(&head, 0);
    SLTPrint(head); // 输出: 0 -> 1 -> 2 -> 3 -> NULL

    // 测试尾删
    SLTPopBack(&head);
    SLTPrint(head); // 输出: 0 -> 1 -> 2 -> NULL

    // 测试头删
    SLTPopFront(&head);
    SLTPrint(head); // 输出: 1 -> 2 -> NULL

    // 测试查找
    SLNode* found = SLTFind(head, 2);
    if (found) {
        printf("找到节点: %d\n", found->val); // 输出: 找到节点: 2
    }
    else {
        printf("未找到节点\n");
    }

    // 测试插入
    SLTInsert(&head, found, 4);
    SLTPrint(head); // 输出: 1 -> 2 -> 4 -> NULL

    // 测试删除
    SLTErase(&head, found);
    SLTPrint(head); // 输出: 1 -> 4 -> NULL

    // 测试销毁
    SLTDestroy(&head);
    SLTPrint(head); // 输出: NULL

    return 0;
}
