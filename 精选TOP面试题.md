# leetcode

## 两数相加

这里使用了一个小技巧，用head一个空的节点来表示头节点，从head节点的后一个节点开始赋值。

``` python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return None
        head=ListNode(0)
        cur=head
        cur1=l1
        cur2=l2
        carry=0
        while cur1 or cur2:
            x=cur1.val if cur1 else 0
            y=cur2.val if cur2 else 0
            num=carry+x+y
            carry=int(num/10)
            cur.next=ListNode(num%10)
            cur=cur.next
            if cur1:
                cur1=cur1.next
            if cur2:
                cur2=cur2.next
        if carry>0:
            cur.next=ListNode(1)
        return head.next
}
```

## 无重复字符的最长子串

类似于DP方法，但是这种方法耗时比较多，循环里套循环了

``` python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        # dp方法，以i结尾的最长子串
        dic={}
        s=list(s)
        dic[0]=[s[0]]
        max_length=1
        for i in range(1,len(s)):
            dic[i]=dic[i-1][:]
            dic[i].append(s[i])
            for j in range(len(dic[i])-1-1,-1,-1):
                if dic[i][j] == s[i]:
                    dic[i]=dic[i][j+1:]
                    break
            max_length=max(max_length,len(dic[i]))
        return max_length
```

滑动窗口法试试，再用空间换时间，使用dic字典来换时间，时间空间都是O(N)的

``` python
class Solution:

    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        # 滑动窗口方法，两个指针，相当于求出了该字符串中的所有的不重复的子串了
        i=0
        j=1
        dic={}
        dic[s[0]]=0
        max_length=1
        while j<=len(s)-1:
            # or dic[s[j]]<i这个很重要！！！
            if s[j] not in dic or dic[s[j]]<i:
                dic[s[j]]=j
                max_length=max(max_length,j-i+1)
                j+=1
            else:
                i=dic[s[j]]+1
                dic[s[j]]=j
                max_length=max(max_length,j-i+1)
                j+=1
        return max_length
```

## 寻找两个有序数组的中位数(困难)

题目中需要的是时间复杂度在O(log(min(m, n)))，这种时间复杂度明显是需要二分查找或者递归的算法，而且min(m, n)提示了

中位数可以首先在第一个数组里面找一个中间的数，比如第i个数，那么在nums2里面有j个数，如果需要划分一刀把两个数字分开的话就一定要左右个数一样，或者左边比右边多一个数。

为了简化代码，不分情况讨论，我们使用一个小trick，我们分别找第 (m+n+1) / 2 个，和 (m+n+2) / 2 个，然后求其平均值即可，这对奇偶数均适用。加入 m+n 为奇数的话，那么其实 (m+n+1) / 2 和 (m+n+2) / 2 的值相等，相当于两个相同的数字相加再除以2，还是其本身。

暂时写了一个不完整的，边界条件需要很多的考虑，我这里考虑的不完全。

``` python
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
            if imin == m-1:
                left = nums1[-1]
                right = nums2[0]
            elif imin == 0:
                left = nums2[-1]
                right = nums1[0]
            else:
                jmin = half_len-(imin+1)-1
                left = max(nums1[imin], nums2[jmin])
                right = min(nums1[imin+1], nums2[jmin+1])
            # print(imin)
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
res = func.findMedianSortedArrays([5, 6, 7, 8], [1, 2, 3, 4])
print(res)

```

## 整数反转

``` python
class Solution:
    def reverse(self, x: int) -> int:
        max_int = pow(2, 31)-1
        min_int = -pow(2, 31)
        res = 0
        flag = 1
        if x < 0:
            x *= -1
            flag = -1
        while x != 0:
            pop = (x % 10)
            x = int(x/10)
            res = res*10+pop
            # print(pop, res, x)
        res *= flag
        if res > max_int or res < min_int:
            return 0
        else:
            return res
```

## 字符串转换整数 (atoi)

中等，就是题目有点长，条件判断我这里应该有点多了，反正需要遍历一个个判断，最好仔细一点，防止越界什么的。

``` python
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
```

## 正则表达式匹配

```python
```
