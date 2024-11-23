class Solution:
    def minSwaps(self , a: List[int], b: List[int]) -> int:
        n = len(a)
        modifications = 0
        for i in range(n):
            if a[i] == b[i]:
                continue
            elif b[i] == a[i]:
                continue
            elif n % 2 == 1 and i == n // 2:
             
                modifications += 1
            elif a[n - 1 - i] == b[i]:
              
                continue
            else:
                modifications += 1
        return modifications
