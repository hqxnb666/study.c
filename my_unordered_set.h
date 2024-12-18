#pragma once
namespace bit
{
   
	template<class K,class Hash = HashFunc<K>>
	class unordered_set
	{
		struct SetKeyOfT
		{
			const K& operator()(const K& key)
			{
				return key;
			}
		};
		typedef typename hash_bucket::HashTable<K, K, SetKeyOfT, Hash>::iterator iterator;
		std::pair<iterator, bool> insert(const K& key)
		{
			return _ht.Insert(key);
		}
		iterator begin()
		{
			return _ht.begin();
		}
		iterator end()
		{
			return _ht.end();
		}
	private:
		hash_bucket::HashTable<K, K,SetKeyOfT,Hash> _ht;
	};
}