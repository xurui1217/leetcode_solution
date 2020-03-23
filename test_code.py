from typing import List


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        if not buildings:
            return []
        res = []
        for k in buildings:
            res.append([k[0], k[2]])
            res.append([k[1], 0])
        res.sort(key=lambda x: x[0])
        return res


func = Solution()
res = func.getSkyline(
    [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]])
print(res)
