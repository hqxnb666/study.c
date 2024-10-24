#pragma once
#include <iostream>
using namespace std;
template<class K>
struct BSTreeNode
{
	BSTreeNode<K>* _left;
	BSTreeNode<K>* _right;
	K _key;

	BSTreeNode(const K& key)
		:_left(nullptr)
		,_right(nullptr)
		,_key(key)
	{}
};

template <class K>
class BSTree {
	typedef BSTreeNode<K> Node;
public:
	bool Insert(const K& key)
	{
		if (_root == nullptr)
		{
			_root = new Node(key);
			return true;
		}
		Node* parent = nullptr;
		Node* cur = _root;
		while (cur)
		{
			if (cur->_key < key)
			{
				parent = cur;
				cur = cur->_right;
			}
			else if (cur->_key > key)
			{
				parent = cur;
				cur = cur->_left;
			}
			else {
				return false;
			}
		}
		cur = new Node(key);
		if (parent->_key > key) {
			parent->_left = cur;
		}
		else {
			parent->_right = cur;
		}
		return true;
	}
	bool Erase(const K& key) {
		Node* parent = nullptr;
		Node* cur = _root;
		while (cur) {
			if (cur->_key < key)
			{
				parent = cur;
				cur = cur->_right;
			}
			else if (cur->_key > key) {
				parent = cur;
				cur = cur->_left;
			}
			else {
				if (cur->_left == nullptr) {
					if (cur == _root) {
						_root = cur->_right;
					}
					else
					{
						if (cur == parent->_left)
						{
							parent->_left = cur->_right;
						}
						else
						{
							parent->_right = cur->_right;
						}
					}
					delete cur;
				}
				else if (cur->_right == nullptr)
				{
					if (cur == _root) {
						_root = cur->_left;
					}
					else
					{
						if (cur == parent->_left)
						{
							parent->_left = cur->_left;
						}
						else
						{
							parent->_right = cur->_left;
						}
					}
					delete cur;
				}
				else
				{
					//左右都不为空
					//替换法删除
					//查找右子树的最左节点代替删除
					Node* rightMinParent = cur;
					Node* rightMin = cur->_right;
					while (rightMin->_left)
					{
						rightMinParent = rightMin;
						rightMin = rightMin->_left;
					}
					swap(cur->_key, rightMin->_key);
					if (rightMinParent->_left == rightMin) {
						rightMinParent->_left = rightMin->_right;
					}
					else {
						rightMinParent->_right = rightMin->_right;
					}
					delete rightMin;
			}
				return true;
	}
	bool Find(const K& key)
	{
		Node* cur = _root;
		while (cur)
		{
			if (cur->_key < key)
			{
				
				cur = cur->_right;
			}
			else if (cur->_key > key)
			{
				
				cur = cur->_left;
			}
			else {
				return true;
			}
		}
		return false;
	}
	void InOrder() {
		_InOrder(_root);
	}
	void _InOrder(Node* root)
	{
		if (root == nullptr) {
			return;
		}
		_InOrder(root->_left);
		cout << root->_key << "";
		_InOrder(root->_right);
	}
private:
	Node* _root = nulalptr;
};

void TestBSTree1()
{
	int a[] = { 8,3,1,10,6,4,7,14,13 };
	BSTree<int> t1;
	for (auto e : a) {
		t1.Insert(e);
	}
	t1.InOrder();
}