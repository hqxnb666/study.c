#include "SList.h"

int main() {
    SLNode* head = NULL;

    // ����β��
    SLTPushBack(&head, 1);
    SLTPushBack(&head, 2);
    SLTPushBack(&head, 3);
    SLTPrint(head); // ���: 1 -> 2 -> 3 -> NULL

    // ����ͷ��
    SLTPushFront(&head, 0);
    SLTPrint(head); // ���: 0 -> 1 -> 2 -> 3 -> NULL

    // ����βɾ
    SLTPopBack(&head);
    SLTPrint(head); // ���: 0 -> 1 -> 2 -> NULL

    // ����ͷɾ
    SLTPopFront(&head);
    SLTPrint(head); // ���: 1 -> 2 -> NULL

    // ���Բ���
    SLNode* found = SLTFind(head, 2);
    if (found) {
        printf("�ҵ��ڵ�: %d\n", found->val); // ���: �ҵ��ڵ�: 2
    }
    else {
        printf("δ�ҵ��ڵ�\n");
    }

    // ���Բ���
    SLTInsert(&head, found, 4);
    SLTPrint(head); // ���: 1 -> 2 -> 4 -> NULL

    // ����ɾ��
    SLTErase(&head, found);
    SLTPrint(head); // ���: 1 -> 4 -> NULL

    // ��������
    SLTDestroy(&head);
    SLTPrint(head); // ���: NULL

    return 0;
}
