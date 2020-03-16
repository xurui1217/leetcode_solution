# -*- coding:utf-8 -*-
import collections


class Solution:
    def letterCombinations(self, digits: str) -> [str]:
        if not digits:
            return []
        dic = {'2': ['a', 'b', 'c'],
               '3': ['d', 'e', 'f'],
               '4': ['g', 'h', 'i'],
               '5': ['j', 'k', 'l'],
               '6': ['m', 'n', 'o'],
               '7': ['p', 'q', 'r', 's'],
               '8': ['t', 'u', 'v'],
               '9': ['w', 'x', 'y', 'z']}
        # res = []
        # for word in dic[digits[0]]:
        #     res.append(word)
        # k = 1
        # while k <= len(digits)-1:
        #     las = []
        #     for cur in res:
        #         for word in dic[digits[k]]:
        #             las.append(cur+word)
        #     res = las[:]
        #     k += 1
        q = collections.deque()
        q.append('')
        for num in digits:
            size = len(q)
            words = dic[num]
            for i in range(size):
                tmp = q.popleft()
                for j in words:
                    q.append(tmp+j)
        return list(q)


func = Solution()
res = func.letterCombinations('23')
print(res)
