# -*- coding:utf-8 -*-
import heapq
import collections
from collections import deque


class Node:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def myAtoi(self, str: str) -> int:
        res = 0
        ch = str
        int_max = 2**31-1
        int_min = -2**31
        i = 0
        if ch == '':
            return 0
        while i <= len(ch)-1 and ch[i] == ' ':
            i += 1
        if i == len(ch):
            return 0
        # print(i)
        # 第一个字符
        flag = 1
        if ch[i] == '+':
            i += 1
            flag = 1
        elif ch[i] == '-':
            i += 1
            flag = -1
        elif ch[i].isdigit():
            flag = 1
        else:
            return 0
        # print(i)
        # 第一个数字
        while i <= len(ch)-1 and ch[i].isdigit():
            res = res*10+int(ch[i])
            i+=1
        res *= flag
        if res>int_max:
            return int_max
        if res<int_min:
            return int_min
        return res


func = Solution()
res = func.myAtoi('42')
print(res)
