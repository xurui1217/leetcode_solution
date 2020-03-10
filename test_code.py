# -*- coding:utf-8 -*-
import heapq
import collections
from collections import deque
import random


class Solution:
    def __init__(self, ch=[]):
        self.ch = ch

    def searchfirstword(self, id1, id2):
        ch1 = self.ch[id1-1]
        ch2 = self.ch[id2-1]
        for i in range(min(len(ch1), len(ch2))):
            if ch1[i] != ch2[i]:
                return i
            else:
                if i == min(len(ch1), len(ch2))-1:
                    return i
        return 0


if __name__ == "__main__":
    stopword = ''
    inp = []
    for line in iter(input, stopword):
        inp.append(line)
    n = int(inp[0])
    ch = [k for k in inp[1:n+1]]
    # print(ch)
    num_search = [k for k in inp[n+1:]]
    # print(num_search)
    func = Solution(ch)
    res = []
    for i in range(len(num_search)):
        search_id = num_search[i].split()
        id1 = int(search_id[0])
        id2 = int(search_id[1])
        # print(ch_search_1, ch_search_2)
        res.append(func.searchfirstword(id1, id2))
    # print(res)
    for k in res:
        print(k)
