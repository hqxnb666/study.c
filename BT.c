#define _CRT_SECURE_NO_WARNINGS
#include "Binary.h"
#include "myQueue.h"


int main()
{
	/*TreeNode* root = CreateNode();
	PrevOrder(root);
	*/




	/*char arr[101];
	scanf("%s", arr);
	int a = 0;
	CreateNode(arr, &a);*/

	TreeNode* root = CreateNode();
	BinaryTreeComplete(root);

	return 0;
}