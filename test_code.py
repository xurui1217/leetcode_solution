# -*- coding:utf-8 -*-
import heapq
import collections
from collections import deque


class Node:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        head = Node(0)
        cur = head
        for i in range(1, n):
            tmp = Node(i)
            cur.next = tmp
            cur = cur.next
        cur.next = head
        cur = head
        # for i in range(5):
        #     print(cur.val)
        #     cur=cur.next
        while cur.next != cur:
            for i in range(m-1):
                cur=cur.next
            print(f'delete:{cur.val}')
            cur.val=cur.next.val
            cur.next=cur.next.next
        return cur.val


func = Solution()
res = func.lastRemaining(5, 3)
print(res)

'''
def partition(data_list, begin, end):
    # 选择最后一个元素作为分区键
    partition_key = data_list[end]

    # index为分区键的最终位置
    # 比partition_key小的放左边，比partition_key 大的放右边
    index = begin
    for i in range(begin, end):
        if data_list[i] < partition_key:
            data_list[i], data_list[index] = data_list[index], data_list[i]
            index += 1

    data_list[index], data_list[end] = data_list[end], data_list[index]
    return index


def find_top_k(data_list, K):
    length = len(data_list)
    begin = 0
    end = length-1
    index = partition(data_list, begin, end)
    while index != length - K:
        if index > length - K:
            end = index-1
            index = partition(data_list, begin, index-1)
        else:
            begin = index+1
            index = partition(data_list, index+1, end)
    return data_list[index]


data_list = [25, 77, 52, 49, 85, 28, 1, 28, 100, 36]
print(data_list)
print(find_top_k(data_list, 7))
print(data_list)
'''
