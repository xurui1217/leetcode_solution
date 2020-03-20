import collections


class Solution:
    def match_need(self, window, need):
        for k in need:
            if need[k] > window[k]:
                return False
        return True

    def minWindow(self, s: str, t: str) -> str:
        left = 0
        right = 0
        n = len(s)
        min_len = n
        window = {}
        need = {}
        res = []
        for i in range(len(s)):
            if s[i] not in window:
                window[s[i]] = 0
        for i in range(len(t)):
            if t[i] not in window:
                return ''
            if t[i] not in need:
                need[t[i]] = 1
            else:
                need[t[i]] += 1
        while right <= n-1:
            window[s[right]] += 1
            if self.match_need(window, need):
                # 改left
                min_len = right-left+1
                res = s[left:right+1]
                while left <= right:
                    left += 1
                    window[s[left-1]] -= 1
                    if not self.match_need(window, need):
                        # left-1到right是满足的最后一个记录一下
                        if right-(left-1)+1 < min_len:
                            min_len = right-(left-1)+1
                            # print(left)
                            res = s[left-1:right+1]
                        right += 1
                        break

            else:
                # 改right
                right += 1
        return res


func = Solution()
res = func.minWindow("a", "a")
print(res)
