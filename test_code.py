from typing import List
from math import ceil


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 从简单到复杂DP
        if not nums:
            return []
        res = [[nums[0]]]
        for i in range(1, len(nums)):
            cur = []
            for word in res:
                print(word)
                for j in range(len(word)):
                    tmp = word[:]
                    tmp.insert(j, nums[i])
                    cur.append(tmp[:])
                tmp = word[:]
                tmp.append(nums[i])
                cur.append(tmp[:])
            res = cur[:]
        return res


func = Solution()
res = func.permute([1, 2, 3])
print(res)
