# -*- coding:utf-8 -*-
import heapq
import collections
from collections import deque


class Node:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        m, n = len(nums1), len(nums2)
        if m == 0 and n == 0:
            return None
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
        if m == 0:
            left = nums2[(n-1)//2]
            right = nums2[(n)//2]
            res = (left+right)/2
            return res
        if (m+n) % 2 == 0:
            imin, imax, half_len = 0, m-1, (m+n)//2
            # 就先看偶数的情况
            while imin < imax:
                i = (imin+imax)//2
                j = half_len-(i+1)-1
                # print(i, j)
                if max(nums1[i], nums2[j]) <= min(nums1[i+1], nums2[j+1]):
                    break
                if nums1[i] > nums2[j+1]:
                    imax = i
                if nums2[j] > nums1[i+1]:
                    imin = i+1
                # print(imin, imax)
            jmin = half_len-(imin+1)-1
            if imin == m-1:
                left = nums1[-1]
                right = nums2[0]
            elif imin == 0:
                left = nums2[-1]
                right = nums1[0]
            else:
                left = max(nums1[imin], nums2[jmin])
                right = min(nums1[imin+1], nums2[jmin+1])
            res = (left+right)/2
        else:
            imin, imax, half_len = 0, m-1, (m+n)//2+1
            # 看奇数的情况
            while imin < imax:
                i = (imin+imax)//2
                j = half_len-(i+1)-1
                # print(i, j)
                if max(nums1[i], nums2[j]) <= min(nums1[i+1], nums2[j+1]):
                    break
                if nums1[i] > nums2[j+1]:
                    imax = i
                if nums2[j] > nums1[i+1]:
                    imin = i+1
                # print(imin, imax)
            jmin = half_len-(imin+1)-1
            res = max(nums1[imin], nums2[jmin])
        return res


func = Solution()
res = func.findMedianSortedArrays([3], [-2, -1])
print(res)
