# -*- coding:utf-8 -*-
from typing import List
import collections


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def reverse(self, start, end):
    pre, cur, nexts = None, start, start
    # 三个指针进行局部翻转
    while cur != end:
        nexts = nexts.next
        # 箭头反指
        cur.next = pre
        # 更新pre位置
        pre = cur
        # 更新cur位置
        cur = nexts
    return pre


class Solution:
    def reverse(self, left, right):
        pre = None
        cur = left
        while cur != right:
            las = cur.next
            cur.next = pre
            pre = cur
            cur = las
        return right, left

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return None
        p_head = ListNode(None)
        p_head.next = head
        pre = p_head
        cur = p_head
        flag = True
        while True:
            # 找到第K个节点，如果出现None节点则已经到了最后阶段，直接设定flag跳出
            for i in range(k):
                if not cur:
                    flag = False
                    break
                cur = cur.next
            if flag == False:
                return p_head.next
            las = cur.next
            pre.next, tmp = self.reverse(pre.next, cur)
            tmp.next = las
            pre = tmp
            cur = pre
        return p_head.next


func = Solution()
res = func.generateParenthesis(3)
print(res)
