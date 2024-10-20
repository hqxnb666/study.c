#define _CRT_SECURE_NO_WARNINGS 1
#include <assert.h>
#include <iostream>
using namespace std;
template<class K, class V>
struct AVLTreeNode
{
	AVLTreeNode<K, V>* _left;
	AVLTreeNode<K, V>* _right;
	AVLTreeNode<K, V>* _parent;
	pair<K, V> _kv;
	int _bf; //balance factor
	AVLTreeNode(const pair<K, V>& _kv)
		:_left(nullptr)
		, _right(nullptr)
		, _parent(nullptr)
		, _kv(_kv)
		, _bf(0)
	{}
};

template<class K, class V>
class AVLTree
{
	typedef AVLTreeNode<K, V> Node;
public:

	bool insert(const pair<K, V>& kv)
	{
		if (_root == nullptr) {
			_root = new Node(kv);
			return true;
		}
		Node* parent = nullptr;
		Node* cur = _root;
		while (cur) {
			if (cur->_kv.first < kv.first) {
				parent = cur;
				cur = cur->_right;
			}
			else if (cur->_kv.first > kv.first) {
				parent = cur;
				cur = cur->_left;
			}
			else
			{
				return false;
			}
		}

		cur = new Node(kv);
		if (parent->_kv.first < kv.first) {
			parent->_right = cur;
		}
		else
		{
			parent->_left = cur;
		}
		cur->_parent = parent;//每次插入都和父亲链接上
		//现在每次插入要更新平衡因子

		while (parent)
		{
			if (cur == parent->_left) {
				parent->_bf--;
			}
			else {
				parent->_bf++;
			}
			if (parent->_bf == 0)
			{
				break;
			}
			else if (parent->_bf == 1 || parent->_bf == -1) {
				//继续向上更新
				cur = parent;
				parent = parent->_parent;
			}
			else if (parent->_bf == 2 || parent->_bf == -2) {
				//当前子树出问题了
				if (parent->_bf == -2 && cur->_bf == -1)
				{
					RotateR(parent);
				}
				else if (parent->_bf == 2 && cur->_bf == 1)
				{
					RotateL(parent);
				}
				else if (parent->_bf == 2 && cur->_bf == -1)
				{
					RotateRL(parent);
				}
				else if (parent->_bf == -2 && cur->_bf == 1)
				{
					RotateLR(parent);
				}
				break;
			}
			else
			{
				assert(false);
			}
		}


		return true;
	}


	
	Node* Find(const K& key)
	{
		Node* cur = _root;
		while (cur)
		{
			if (cur->_kv.first < key)
			{
				cur = cur->_right;
			}
			else if (cur->_kv.first > key)
			{
				cur = cur->_left;
			}
			else
			{
				return cur;
			}
		}
		return nullptr;
	}

	

	void RotateR(Node* parent)
	{
		Node* subL = parent->_left;
		Node* subLR = subL->_right;
		parent->_left = subLR;
		if(subLR)
		subLR->_parent = parent;
		subL->_right = parent;

		Node* ppNode = parent->_parent;
		parent->_parent = subL;
		if (parent == _root) {
			_root = subL;
			subL->_parent = nullptr;
		}
		else
		{
			if (ppNode->_left == parent) {
				ppNode->_left = subL;
			}
			else
			{
				ppNode->_right = subL;
			}
			subL->_parent = ppNode;
		}
		parent->_bf = subL->_bf = 0;
	}


	void RotateL(Node* parent)
	{
		Node* subR = parent->_right;
		Node* subRL = subR->_left;

		parent->_right = subRL;
		if (subRL)
			subRL->_parent = parent;
		subR->_left = parent;

		Node* ppNode = parent->_parent;
		parent->_parent = subR;
		if (parent == _root) {
			_root = subR;
			subR->_parent = nullptr;
		}
		else
		{
			if (ppNode->_left == parent) {
				ppNode->_left = subR;
			}
			else
			{
				ppNode->_right = subR;
			}
			subR->_parent = ppNode;
			
		}
		parent->_bf = subR->_bf = 0;
		
	}
	void RotateLR(Node* parent)
	{
		Node* subL = parent->_left;
		Node* subLR = subL->_right;

		int bf = subLR->_bf;
		RotateL(parent->_left);
		RotateR(parent);
		//单旋结束后平衡因子会记为0，实际上平衡因子会改变
		if (bf == -1)//代表在左边插入
		{
			subLR->_bf = 0;
			subL->_bf = 0;
			parent->_bf = 1;
		}
		else if (bf == 1)
		{
			subLR->_bf = 0;
			subL->_bf = -1;
			parent->_bf = 0;
		}
		else if (bf == 0)
		{
			subLR->_bf = 0;
			subL->_bf = 0;
			parent->_bf = 0;
		}
		else {
			assert(false);
			//理论上不可能，万一写错了呢
		}
	}
	void RotateRL(Node* parent)
	{
		Node* subR = parent->_right;
		Node* subRL = subR->_left;
		int bf = subRL->_bf;


		RotateR(subR);
		RotateL(parent);
		if (bf == -1)
		{
			parent->_bf = 0;
			subRL->_bf = 0;
			subR->_bf = 1;
		}
		else if (bf == 1)
		{
			parent->_bf = -1;
			subRL->_bf = 0;
			subR->_bf = 0;
		}
		else if (bf == 0)
		{
			parent->_bf = 0;
			subRL->_bf = 0;
			subR->_bf = 0;
		}
		else
		{
			assert(false);
		}
	}
	void InOrder() {
		_InOrder(_root);
	}

	bool IsBalance()
	{
		return _IsBalance(_root);
	}
private:
	void _InOrder(Node* root)
	{
		if (root == nullptr) {
			return;
		}
		_InOrder(root->_left);
		cout << root->_kv.first << " " << root->_kv.second << endl;
		_InOrder(root->_right);
	}
	int _Height(Node* root)
	{
		if (root == nullptr)
		{
			return 0;
		}
		return max(_Height(root->_left), _Height(root->_right)) + 1;
	}
	bool _IsBalance(Node* root)
	{
		if (root == nullptr)
		{
			return true;
		}
		int leftHeight = _Height(root->_left);
		int rightHeight = _Height(root->_right);
		if (abs(leftHeight - rightHeight) >= 2)
			return false;
		return _IsBalance(root->_left) && _IsBalance(root->_right);
	}
private:
	Node* _root = nullptr;
};
