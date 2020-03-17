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

有几种方法，回溯，动态规划，一般用DP方法，设定dp[i][j]为s[i]和p[j]结尾的序列是不是满足匹配，有0或者1两种状态。

看转移方程：已知dp[i-1][j-1]的值，计算dp[i][j]的值，需要看s[i]和p[j]

1. s[i]==p[j] or p[j]=='.'这样的话dp[i][j]=dp[i-1][j-1]，很明显可以匹配
2. p[j]=='*'这种情况是匹配之前一个字符的重复，所以需要看p[j-1]和s[i]的关系又可以分为

p[j-1]!=s[i]，那么相当于这个'*'是没有匹配到重复的字符，dp[i][j]=dp[i][j-2]

p[j-1]==s[i] or p[j-1]=='.'，那么应该是找到了重复的字符，当然也可以额把它去掉当做没有重复字符=dp[i][j-2]，或者dp[i][j]=dp[i-1][j]

实际写代码的时候这种题目经常会出现string越界的问题，或者是string的空很难判断的问题，可以用前l个s字符和前r个p字符，然后对应的index为i=l-1，j=r-1，循环的话可以从1开始循环。

``` python
# -*- coding:utf-8 -*-

class Solution:
    def isMatch(self, s: str, p: str):
        if not p:
            return not s
        if not s and len(p) == 1:
            return False
        m, n = len(s), len(p)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = 1
        dp[0][1] = 0
        for c in range(2, n+1):
            j = c-1
            if p[j] == '*':
                dp[0][c] = dp[0][c-2]
        for r in range(1, m+1):
            i = r-1
            for c in range(1, n+1):
                j = c-1
                if s[i] == p[j] or p[j] == '.':
                    dp[r][c] = dp[r-1][c-1]
                elif p[j] == '*':
                    if p[j-1] == s[i] or p[j-1] == '.':
                        dp[r][c] = dp[r-1][c] or dp[r][c-2]
                    else:
                        dp[r][c] = dp[r][c-2]
                else:
                    dp[r][c] = 0
        print(dp)
        return True if dp[m][n] else False

func = Solution()
res = func.isMatch(s='abb', p='ab*')
# print(res)
```

## 盛最多水的容器

感觉是求一个子序列中的最大值和第二大的值的差值，使得这样的差值最大，并且最大值和第二大的值需要在子序列的左右两端。（错误的看题）

正确思路是，只看左右指针的值，然后看较小的值，再乘一个有指针到左指针的长度，就得到了水的面积（求最大面积）。

根据官方发布的解题步骤：以首末两点为起点，较短的那一根向内侧移动，直到两指针相遇。要证明这种方法的正确性，只需要证明该方法得到的面积的移动轨迹经过最大面积。

一般DP或者双指针遍历，这里双指针把，dp应该不行把

``` py
# -*- coding:utf-8 -*-

class Solution:
    def maxArea(self, height) -> int:
        i = 0
        j = len(height)-1
        max_volume = 0
        while i < j:
            if height[i] <= height[j]:
                max_volume = max(max_volume, height[i]*(j-i))
                i += 1
            else:
                max_volume = max(max_volume, height[j]*(j-i))
                j -= 1
        return max_volume

func = Solution()
res = func.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7])
```

## 罗马数字转整数

hash表，关键是判断是否一个当前字符后面的一个字符比这个字符大，就会需要转换一下值的大小。

``` py
# -*- coding:utf-8 -*-
class Solution:
    def romanToInt(self, s: str) -> int:
        if not s:
            return 0
        dic = {'I': 1,
               'V': 5,
               'X': 10,
               'L': 50,
               'C': 100,
               'D': 500,
               'M': 1000}
        i = 0
        count = 0
        while i <= len(s)-1:
            if i <= len(s)-2 and dic[s[i]] < dic[s[i+1]]:
                count += dic[s[i+1]]-dic[s[i]]
                i += 2
            else:
                count += dic[s[i]]
                i += 1
        return count

func = Solution()
res = func.romanToInt("III")
# print(res)
```

## 三数之和

直接暴力遍历三个循环，最后需要去除重复的数字，但是会超出时间，不能通过。

``` py
# -*- coding:utf-8 -*-
class Solution:
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        nums.sort()
        # print(nums)
        res = []
        for i in range(len(nums)-2):
            for j in range(i+1, len(nums)-1):
                for k in range(j+1, len(nums)):
                    if nums[i]+nums[j]+nums[k] == 0:
                        if [nums[i], nums[j], nums[k]] not in res:
                            res.append([nums[i], nums[j], nums[k]])
        return res

func = Solution()
res = func.threeSum([-1, 0, 1, 2, -1, -4])
```

优化再优化，考虑其他的办法，双指针，但是需要用到一些数学上的知识，先快速排序O(NlogN)，首先选定第一个值比如为s[k], 第二个值和第三个值为s[i]和s[j]，如果sum==0则i+=1再j-=1, 如果sum>0则j-=1, 如果sum<0则i+=1, 中间需要跳过已经出现过的值，直接去向下一个没有出现过的值。

能通过leetcode了，但是复杂度感觉仍然很高，用时击败5.06%醉了。

``` py
# -*- coding:utf-8 -*-
class Solution:
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        nums.sort()
        # print(nums)
        res = []
        k = 0
        while k <= len(nums)-3:
            i = k+1
            j = len(nums)-1
            while i < j:
                # print(k, i, j)
                if nums[k]+nums[i]+nums[j] == 0:
                    res.append([nums[k], nums[i], nums[j]])
                    while i < j and nums[i+1] == nums[i]:
                        i += 1
                    i += 1
                    while i < j and nums[j-1] == nums[j]:
                        j -= 1
                    j -= 1
                elif nums[k]+nums[i]+nums[j] > 0:
                    while i < j and nums[j-1] == nums[j]:
                        j -= 1
                    j -= 1
                elif nums[k]+nums[i]+nums[j] < 0:
                    while i < j and nums[i+1] == nums[i]:
                        i += 1
                    i += 1
                # print(k, i, j)
            while k <= len(nums)-3 and nums[k+1] == nums[k]:
                k += 1
            k += 1
        return res

func = Solution()
res = func.threeSum([-1, 0, 1, 2, -1, -4])
```

## 电话号码的字母组合

python中字符和asc码的转换过程

``` py
sum = ord('A')
# 结果为65
sum = chr(65)
# 结果为A
```

结果发现7和9不是按照规则来的，是4个数，重写一下呗

时间复杂度应该挺高的，每一个循环内部还需要再循环，类似于队列的形式

``` py
# -*- coding:utf-8 -*-
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
        res = []
        for word in dic[digits[0]]:
            res.append(word)
        k = 1
        while k <= len(digits)-1:
            las = []
            for cur in res:
                for word in dic[digits[k]]:
                    las.append(cur+word)
            res = las[:]
            k += 1
        return res

func = Solution()
res = func.letterCombinations('23')
# print(res)
```

看题解之后，尝试用纯队列写一个版本，这不就是和我上面写的一毛一样嘛，时间用的都一样

``` py
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

```

## 删除链表的倒数第N个节点

双指针从前往后走，第二个指针走到结尾，第一个指针为倒数第n个节点。

注意考虑边界条件，最前面，最后面，中间情况，我这里时间为O(N), 空间O(1), 一次遍历

``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return None
        if n<=0:
            return head
        # 判断是否n越界了，题目中有n一定有效，那么可以不用判断
        # 一趟扫描也可以
        l=head
        r=head
        for i in range(n):
            r=r.next
        while r:
            l=l.next
            r=r.next
        if l.next:
            l.val=l.next.val
            l.next=l.next.next
        else:
            if l==head:
                return None
            cur=head
            while cur.next.next:
                cur=cur.next
            cur.next=None
        return head
```

## 有效的括号

写过了，栈

``` py
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s)%2!=0:
            return False
        helper=[]
        dic={'(':')','[':']','{':'}'}
        for i in range(len(s)):
            if s[i] in ['(','[','{']:
                helper.append(s[i])
            else:
                if helper==[]:
                    return False
                ch=helper.pop()
                if s[i]!=dic[ch]:
                    return False
        if helper!=[]:
            return False
        else:
            return True
```

## 括号生成

几种办法：DFS，BFS，回溯，动态规划

DFS看的别人的

``` py
# -*- coding:utf-8 -*-
from typing import List
import collections

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:

        res = []
        cur_str = ''

        def dfs(cur_str, left, right):
            """
            :param cur_str: 从根结点到叶子结点的路径字符串
            :param left: 左括号还可以使用的个数
            :param right: 右括号还可以使用的个数
            :return:
            """
            if left == 0 and right == 0:
                res.append(cur_str)
                return
            if left>right:
                return
            if left > 0:
                dfs(cur_str + '(', left - 1, right)
            if right > 0:
                dfs(cur_str + ')', left, right - 1)

        dfs(cur_str, n, n)
        return res

func = Solution()
res = func.generateParenthesis(3)
print(res)
```

动态规划，先求n-1对括号的生成情况，然后再加一对括号看是什么情况。

当我们清楚所有 i<n 时括号的可能生成排列后，对与 i=n 的情况，我们考虑整个括号排列中最左边的括号。

它一定是一个左括号，那么它可以和它对应的右括号组成一组完整的括号 "( )"，我们认为这一组是相比 n-1 增加进来的括号。

``` py
# -*- coding:utf-8 -*-
from typing import List
import collections

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n <= 0:
            return []
        res = [['']]
        res.append(['()'])
        if n == 1:
            return res[1]
        for i in range(2, n+1):
            cur = []
            for j in range(i):
                pre = res[j]
                las = res[i-1-j]
                for k1 in pre:
                    for k2 in las:
                        fin = '('+k1+')'+k2
                        cur.append(fin)
            res.append(cur)
        return res[n]

func = Solution()
res = func.generateParenthesis(3)
print(res)
```

## 合并K个排序链表

合并k个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

暴力法，直接全部遍历一遍O(N)，然后再排序O(NlogN)，再遍历进节点O(N), 空间就N个节点保存值O(N)。

``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        res = []
        for k_head in lists:
            while k_head:
                res.append(k_head.val)
                k_head = k_head.next
        res.sort()
        head = ListNode(None)
        cur = head
        for i in range(len(res)):
            cur.next = ListNode(res[i])
            cur = cur.next
        return head.next
```

分治法，两两合并，然后再两两合并

## 两两交换链表中的节点

设定一个虚拟头结点p_head, 然后用pre，cur，las表示是三个节点，用python的牛逼链表节点转移大法：

``` py
pre.next, cur.next, las.next=cur.next, las.next, cur
```

``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head:
            return None
        p_head=ListNode(None)
        p_head.next=head
        pre=p_head
        cur=head
        while cur and cur.next:
            las=cur.next
            pre.next,cur.next,las.next=cur.next,las.next,cur
            pre=cur
            cur=cur.next
        return p_head.next
```

## K 个一组翻转链表

和上面的题目差不多，但是翻转的个数是不确定的，所以肯定是要一些list操作，先把头节点和尾部的节点放在list里，从list里从后往前串起来，再把头尾接上去。

注意边界条件也是需要判断好的。我这里重新谢了一个函数来表示K链表的翻转，返回头结点和尾节点，注意翻转链表需要三个指针，per, cur, las一开始需要设置一个空的头节点指针。

``` py
class Solution:
    def reverse(self, left, right):
        pre = None
        cur = left
        while cur != right:
            las = cur.next
            cur.next = pre
            pre = cur
            cur = las
        cur.next=pre
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
                cur = cur.next
                if not cur:
                    flag = False
                    break
            if flag == False:
                return p_head.next
            las = cur.next
            pre.next, tmp = self.reverse(pre.next, cur)
            tmp.next = las
            pre = tmp
            cur = pre
        return p_head.next
```

