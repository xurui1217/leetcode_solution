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

## 两数相除

两数相除不能用乘法和除法，只能用加减法和移位运算，移位也就是只能乘2。

主要思想是除数一直左移位，代表×2，但是不能超过被除数，并且记录左移的次数代表×了多少个2，这里也直接通过1左移位的方式直接记录了乘多少了，不用再计算次数。

考察了移位符号的运算。

``` py
# -*- coding:utf-8 -*-
class Solution:
    def divide(self, dividend, divisor):
        if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0):
            flag = 1
        else:
            flag = -1
        dividend = abs(dividend)
        divisor = abs(divisor)
        res = 0
        while dividend >= divisor:
            tmp_divisor, count = divisor, 1
            while dividend >= tmp_divisor:
                count <<= 1
                tmp_divisor <<= 1
                if dividend < tmp_divisor:
                    count >>= 1
                    tmp_divisor >>= 1
                    dividend -= tmp_divisor
                    res += count
                    break
        if flag==-1:
            res=-res
        # 这里的判断是否溢出用移位来判断，[-(2<<30),2<<30-1]，减少很多运算量
        if res > (2 << 30)-1:
            return (2 << 30)-1
        elif res < 0-(2 << 30):
            return 0-(2 << 30)
        return res

func = Solution()
res = func.divide(7, -3)
```

## 旋转搜索排序数组

中间有一个判断是否是在mid左边还是在mid右边，注意二分法的思想，先求一个mid判断这个mid是不是要找到，不是再找[l, mid-1]或者[mid+1.r]

中间有一个判断非常重要 if nums[i] <= nums[mid] 中的等于号

``` py
# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        i, j = 0, len(nums)-1
        while i <= j:
            mid = (i+j)//2
            if nums[mid] == target:
                return mid
            if nums[i] <= nums[mid]:  # left
                if nums[i] <= target < nums[mid]:
                    j = mid-1
                else:
                    i = mid+1
            else:  # right
                if nums[mid] < target <= nums[j]:
                    i = mid+1
                else:
                    j = mid-1
        return -1

func = Solution()
res = func.search([5, 1, 2, 3, 4], 1)
print(res)

```

## 在排序数组中查找元素的第一个和最后一个位置

依旧是万年不变的二分法，两个函数，第一个找开头，第二个找结尾，算好mid代表的意思，以及(l+r)//2代表的含义，有的时候永远也取不到r，所以需要用一个math.ceil(x)向上取整。

``` py
from typing import List
import math

class Solution:
    def searchfirst(self, nums, target):
        l = 0
        r = len(nums)-1
        if nums[0] == target:
            return 0
        while l <= r:
            mid = (l+r)//2
            # print(l, mid, r)
            if nums[mid] == target and nums[mid-1] != target:
                return mid
            if nums[mid] < target:
                l = mid+1
            elif nums[mid] > target:
                r = mid-1
            else:
                r = mid
        return -1

    def searchlast(self, nums, target):
        l = 0
        r = len(nums)-1
        if nums[r] == target:
            return r
        while l <= r:
            mid = math.ceil((l+r)/2)
            # print(l, mid, r)
            if nums[mid] == target and nums[mid+1] != target:
                return mid
            if target < nums[mid]:
                r = mid-1
            elif target > nums[mid]:
                l = mid+1
            else:
                l = mid
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        left = self.searchfirst(nums, target)
        # print(left)
        right = self.searchlast(nums, target)
        # print(right)
        return [left, right]

func = Solution()
res = func.searchRange([5, 7, 7, 8, 8, 10], 8)
# print(res)
```

## 有效的数独

思路：用多个dic来保证速度，只需要一次遍历即可，时间复杂度为O(n^2), n为几行

``` py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        lis_row=[]
        lis_col=[]
        lis_mini=[]
        for i in range(9):
            tmp1,tmp2,tmp3={},{},{}
            lis_row.append(tmp1)
            lis_col.append(tmp2)
            lis_mini.append(tmp3)
        for i in range(9):
            for j in range(9):
                x=board[i][j]
                if x != '.':
                    if x in lis_row[i]:
                        return False
                    else:
                        lis_row[i][x]=1
                    if x in lis_col[j]:
                        return False
                    else:
                        lis_col[j][x]=1
                    id_mini=(i//3)*3+j//3
                    if x in lis_mini[id_mini]:
                        return False
                    else:
                        lis_mini[id_mini][x]=1
        return True
```

## 外观数列

简单题，考虑的好一点即可

``` py
class Solution:
    def countAndSay(self, n: int) -> str:
        pre='1'
        if n==1:
            return pre
        for i in range(2,n+1):
            l=0
            r=0
            count=0
            res=''
            while r<=len(pre)-1:
                count+=1
                r+=1
                if r==len(pre):
                    res+=str(count)+pre[l]
                    break
                if pre[r]!=pre[l]:
                    res+=str(count)+pre[l]
                    count=0
                    l=r
            pre=res
        return res
```

## 接雨水

只能想到暴力法，把当前每一个块的能够积蓄水的高度求出来，然后加起来，O(n^2)

用空间换时间，两次遍历，一遍找到max_left存起来，一遍找到max_right存起来，然后在依次遍历求res

``` py
# 暴力法，超时
class Solution:
    def trap(self, height: List[int]) -> int:
        n=len(height)
        res=0
        for i in range(1,n-1):
            max_left=height[i]
            for j in range(i):
                max_left=max(max_left,height[j])
            max_right=height[i]
            for j in range(j+1,n):
                max_right=max(max_right,height[j])
            res+=min(max_left,max_right)-height[i]
        return res
```

用dic降低时间复杂度，降到了O(N)

``` py
# 用dic保存left，right减少时间
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height: # 这个判断是一定要写的
            return 0
        n=len(height)
        res=0
        dic_left={}
        dic_right={}
        dic_left[0]=height[0]
        dic_right[n-1]=height[n-1]
        for i in range(1,n):
            dic_left[i]=max(dic_left[i-1],height[i])
        for i in range(n-2,0,-1):
            dic_right[i]=max(dic_right[i+1],height[i])
        for i in range(1,n-1):
            res+=min(dic_left[i],dic_right[i])-height[i]
        return res
```

## 全排列

``` py
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
```

## 字母异位词分组

用dic加速，空间换时间

``` py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic={}
        res=[]
        for i in range(len(strs)):
            strs_key=''.join(sorted(strs[i]))
            if strs_key not in dic:
                dic[strs_key]=[strs[i]]
            else:
                dic[strs_key].append(strs[i])
        for k in dic:
            res.append(dic[k])
        return res
```

## Pow(x, n)快速幂

递归，时间复杂度为O(logN)

不用递归直接暴力是O(N)

``` py
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n<0:
            n = -n
            return 1/self.help_(x,n)
        return self.help_(x,n)

    def help_(self,x,n):
        if n==0:
            return 1
        if n%2 == 0:
             #如果是偶数
            return self.help_(x*x, n//2)
        return self.help_(x*x,(n-1)//2)*x
```

## 最大子序和

一看就是老DP了，很快的写一下

``` py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return None
        n=len(nums)
        dp=[0 for _ in range(n)]
        dp[0]=nums[0]
        sum_max=dp[0]
        for i in range(1,n):
            if dp[i-1]>=0:
                dp[i]=dp[i-1]+nums[i]
            else:
                dp[i]=nums[i]
            sum_max=max(sum_max,dp[i])
        return sum_max
```

## 螺旋矩阵

剑指offer中的一个题目，需要考虑几种情况，我的方法是选定坐上角和右下角的元素，确定一个矩形框，然后在for循环取值出来。

基本思想应该挺好理解的，画个图就能说清楚。

``` py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # 按层次来
        if not matrix:
            return []
        n=len(matrix)
        m=len(matrix[0])
        i1=0
        j1=0
        i2=n-1-i1
        j2=m-1-j1
        res=[]
        while i1<=i2 and j1<=j2:
            if i1==i2:
                for j in range(j1,j2+1):
                    res.append(matrix[i1][j])
                break
            if j1==j2:
                for i in range(i1,i2+1):
                    res.append(matrix[i][j1])
                break
            for j in range(j1,j2):
                res.append(matrix[i1][j])
            for i in range(i1,i2):
                res.append(matrix[i][j2])
            for j in range(j2,j1,-1):
                res.append(matrix[i2][j])
            for i in range(i2,i1,-1):
                res.append(matrix[i][j1])
            i1+=1
            j1+=1
            i2-=1
            j2-=1
        return res
```

## 跳跃游戏

应该是DP，想一下怎么搞出来，理论上应该是dp[i]表示该位置能够到达，首先dp[0]=1边界

转移方程咋写，感觉是需要一波操作，首先看如果dp[i]==1, 那么nums[i]代表能够跳到nums[0+nums[0]]的地方，这样的话dp[i:i+nums[i]]全部改为1，可以逐步往后

下面的写法超时了

``` py
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if not nums:
            return False
        n=len(nums)
        dp=[0 for _ in range(n)]
        dp[0]=1
        for i in range(n):
            if dp[i]==1:
                for j in range(nums[i]+1):
                    if i+j<=n-1:
                        dp[i+j]=1
                    else:
                        break
            # print(dp)
        return True if dp[-1]==1 else False
```

看到有个C++的代码用的相同思路，但是其实不需要dp，只需要保存下来每一个点能够到达的最远点即可，这个想法神曲

``` py
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if not nums:
            return False
        n=len(nums)
        k=0
        for i in range(n):
            if i>k:
                return False
            k=max(k,i+nums[i])
        return True
```

## 颜色分类

这个题中，数组只有三种元素：0 1 2，0总是放在最前，2总是放在最后，所以我们只要将0和2的位置放对，那么整个数组自然就有序了

left表示左端0的位置，righ表示右端2的位置；cur遍历数组的索引
如果nums[cur] == 0，我们只需要将当前元素和left表示的元素进行交换，并将left往后移动一位
同理，如果nums[cur] == 1，我们只需要将当前元素和right表示的元素进行交换，并将right往前移动一位
当 cur == right的时候，便结束循环

三个指针法，left表示0的位置，right表示2的位置

``` C
int left = 0, right = nums.size() - 1, cur = 0;
while(cur <= right){

    if(nums[cur] == 0){
        swap(nums[left++], nums[cur++]);
    }else if(nums[cur] == 2){
        swap(nums[right--], nums[cur]);
    }else{
        cur += 1;
    }

}
```

``` py
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # dic=collections.Counter(nums)
        # res=[0]*dic[0]+[1]*dic[1]+[2]*dic[2]
        # nums[:]=res[:]
        cur=0
        left=0
        right=len(nums)-1
        while cur<=right:
            if nums[cur]==0:
                nums[cur],nums[left]=nums[left],nums[cur]
                cur+=1
                left+=1
            elif nums[cur]==2:
                nums[cur],nums[right]=nums[right],nums[cur]
                right-=1
            else:
                cur+=1
```

## 最小覆盖子串

滑动窗口的一般解决方法，设定一个left=0和right=0，先right+=1，然后等到match了之后保存一下res，这个时候再不断left+=1，直到不能match，保存最后能够match的res。

``` py
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
                if right-left+1 <= min_len:
                    min_len = right-left+1
                    # print(left)
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
        if res==[]:
            return ''
        else:
            return res

func = Solution()
res = func.minWindow("a", "a")
# print(res)
```

## 子集

一般这种求枚举的我都是从小到大依次加元素进去，注意import copy然后用deepcopy()方法可以好一点，不要在这个地方出错就尴尬了。

``` py
import collections
from typing import List
import copy

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return []
        res = []
        res.append([])
        for i in range(len(nums)):
            cur = copy.deepcopy(res)
            # print(i, cur)
            for j in range(len(cur)):
                cur[j].append(nums[i])
            res.extend(cur)
            # print(res)
        return res

func=Solution()
res=func.subsets([1, 2, 3])
# print(res)
```

## 单词搜索

思路，DFS回溯法，一般遇到这种题目需要来一个mark来标记已经走过的位置，或者直接在原来的矩阵中标记走过的路径。这里用回溯法的一个坑，如果在一个方向上没有找到有用的，就回过来mark改为0！这一点很重要。

``` py
import collections
from typing import List
import copy

class Solution:
    def DFS(self, board, i, j, mark, word):
        if word == '':
            return True
        direct = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        row = len(board)
        col = len(board[0])
        for dir in range(4):
            cur_i = i+direct[dir][0]
            cur_j = j+direct[dir][1]
            if cur_i < 0 or cur_i >= row or cur_j < 0 or cur_j >= col or mark[cur_i][cur_j] == 1:
                continue
            else:
                if board[cur_i][cur_j] == word[0]:
                    mark[cur_i][cur_j] = 1
                    if self.DFS(board, cur_i, cur_j, mark, word[1:]):
                        return True
                    else:
                        mark[cur_i][cur_j] = 0

    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board:
            return False
        row = len(board)
        col = len(board[0])
        mark = [[0 for _ in range(col)] for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if board[i][j] == word[0]:
                    mark[i][j] = 1
                    if self.DFS(board, i, j, mark, word[1:]):
                        return True
                    else:
                        mark[i][j] = 0

        return False

# func = Solution()
# board = [
#     ['A', 'B', 'C', 'E'],
#     ['S', 'F', 'C', 'S'],
#     ['A', 'D', 'E', 'E']
# ]
# res = func.exist(board, "ABCB")
# print(res)
```

## 删除排序数组中的重复项 II

双指针，搞清楚什么时候覆盖，不是交换！！！

``` py
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # 第一反应双指针
        count=1
        right=1
        left=1
        while right<=len(nums)-1:
            if nums[right] == nums[right - 1]:
                count += 1
            else:
                count=1
            if count<=2:
                nums[left]=nums[right]
                left+=1
            right+=1
        return left
```

## 删除排序链表中的重复元素 II

老双指针了，删除元素的时候判断一下该元素后面有没有元素，有元素的话是不是和该元素的值一毛一样，一样的话就删除后面所有一样的节点，再把自己删掉，不一样就跳过。

``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 三指针
        if not head:
            return None
        PHEAD=ListNode(None)
        pre=PHEAD
        pre.next=head
        cur=head
        while cur:
            las=cur.next
            if las and cur.val==las.val:
                # 不断删除后面一个节点如果是相同的值的话，并且需要把这个节点也去掉
                while cur.next and cur.val==cur.next.val:
                    cur.next=cur.next.next
                # cur此时为单独的值再把这个值去掉即可
                pre.next=cur.next
                cur=cur.next
            else:
                pre=pre.next
                cur=cur.next
        return PHEAD.next
```

## 柱状图中最大的矩形

分治算法，有一点暴力

``` py
import collections
from typing import List
import copy

class Solution:
    def __init__(self):
        self.vol = 0

    def splitrectanglge(self, heights, l, r):
        if l == r:
            self.vol = max(self.vol, heights[l])
        else:
            min_id = -1
            min_high = float('inf')
            for i in range(l, r+1):
                if heights[i] < min_high:
                    min_high = heights[i]
                    min_id = i
            self.vol=max(self.vol,heights[min_id]*(r-l+1))
            if min_id == l:
                self.splitrectanglge(heights, min_id+1, r)
            elif min_id == r:
                self.splitrectanglge(heights, l, min_id-1)
            else:
                self.splitrectanglge(heights, min_id+1, r)
                self.splitrectanglge(heights, l, min_id-1)

    def largestRectangleArea(self, heights: List[int]) -> int:
        # 分治算法
        self.splitrectanglge(heights, 0, len(heights)-1)
        return self.vol

func = Solution()
res = func.largestRectangleArea([2, 1, 5, 6, 2, 3])
print(res)
```

想办法优化这个算法, 看了一个直接return的思路。都会超时

``` py
import collections
from typing import List
import copy

class Solution:
    def splitrectanglge(self, heights, l, r):
        if l > r:
            return 0
        min_id = l
        for i in range(l, r+1):
            if heights[min_id] > heights[i]:
                min_id = i
        left = self.splitrectanglge(heights, l, min_id-1)
        right = self.splitrectanglge(heights, min_id+1, r)
        mid = heights[min_id]*(r-l+1)
        return max(max(left, right), mid)

    def largestRectangleArea(self, heights: List[int]) -> int:
        # 分治算法
        return self.splitrectanglge(heights, 0, len(heights)-1)

func = Solution()
res = func.largestRectangleArea([2, 1, 5, 6, 2, 3])
print(res)
```

优化版本的思路，非单调递增栈！不会

## 岛屿数量

DFS，注意python是写在函数里面的，调用的都是全局的变量

``` py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # DFS
        if not grid:
            return 0
        row=len(grid)
        col=len(grid[0])
        def dfs(i,j):
            grid[i][j]='0'
            direction=[[-1,0],[0,1],[1,0],[0,-1]]
            for k in range(4):
                cur_i=i+direction[k][0]
                cur_j=j+direction[k][1]
                if cur_i<0 or cur_i>=row or cur_j<0 or cur_j>=col:
                    continue
                else:
                    if grid[cur_i][cur_j]=='1':
                        dfs(cur_i,cur_j)
        count=0
        for i in range(row):
            for j in range(col):
                if grid[i][j]=='1':
                    count+=1
                    dfs(i,j)
        return count
```

## 课程表

BFS

算法流程：
统计课程安排图中每个节点的入度，生成 入度表 indegrees。
借助一个队列 queue，将所有入度为 00 的节点入队。
当 queue 非空时，依次将队首节点出队，在课程安排图中删除此节点 pre：
并不是真正从邻接表中删除此节点 pre，而是将此节点对应所有邻接节点 cur 的入度 -1−1，即 indegrees[cur] -= 1。
当入度 -1−1后邻接节点 cur 的入度为 00，说明 cur 所有的前驱节点已经被 “删除”，此时将 cur 入队。
在每次 pre 出队时，执行 numCourses--；
若整个课程安排图是有向无环图（即可以安排），则所有节点一定都入队并出队过，即完成拓扑排序。换个角度说，若课程安排图中存在环，一定有节点的入度始终不为 00。
因此，拓扑排序出队次数等于课程个数，返回 numCourses == 0 判断课程是否可以成功安排。

``` py
from collections import deque

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegrees = [0 for _ in range(numCourses)]
        adjacency = [[] for _ in range(numCourses)]
        queue = deque()
        # Get the indegree and adjacency of every course.
        for cur, pre in prerequisites:
            indegrees[cur] += 1
            adjacency[pre].append(cur)
        # Get all the courses with the indegree of 0.
        for i in range(len(indegrees)):
            if not indegrees[i]: queue.append(i)
        # BFS TopSort.
        while queue:
            pre = queue.popleft()
            numCourses -= 1
            for cur in adjacency[pre]:
                indegrees[cur] -= 1
                if not indegrees[cur]: queue.append(cur)
        return not numCourses
```

DFS

``` py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(i, adjacency, flags):
            # 没有环输出True
            if flags[i] == -1: return True
            if flags[i] == 1: return False
            flags[i] = 1
            for j in adjacency[i]:
                if dfs(j, adjacency, flags)==False:
                    return False
            flags[i] = -1
            return True

        adjacency = [[] for _ in range(numCourses)]
        flags = [0 for _ in range(numCourses)]
        for cur, pre in prerequisites:
            adjacency[pre].append(cur)
        for i in range(numCourses):
            if not dfs(i, adjacency, flags): return False
        return True
```

图的邻接表表示法：

``` py
#图的邻接链表表示法
graph = {'A': ['B', 'C'],
        'B': ['C', 'D'],
        'C': ['D'],
        'D': ['C','G','H'],
        'E': ['F'],
        'F': ['C']}
```

也可以加上边的长度也可以

``` py
#图的邻接链表表示法
graph = {'A': [['B',1], ['C',2]],
        'B': [['C',2], ['D',4]]}
```

拓扑排序，要看一个节点的入度，一直把入度为0的节点提取出来，然后把这些点的边的出口的入度-1然后迭代。

``` py
#遍历图中所有顶点，按照遍历顺序将顶点添加到列表中
vertex = []
def dfs(v):
    if v not in graph:
        return
    for vv in graph[v]:
        if vv not in vertex:
            vertex.append(vv)
            dfs(vv)

for v in graph:
    if v not in vertex:
        vertex.append(v)
        dfs(v)
print(vertex)
```

``` py
#从图中找出从起始顶点到终止顶点的所有路径
import copy

def find_path_all(curr, end, path):
    '''
    :param curr: 当前顶点
    :param end: 要到达的顶点
    :param path: 当前顶点的一条父路径
    :return:
    '''
    if curr == end:
        path_tmp = copy.deepcopy(path)
        path_all.append(path_tmp)
        return
    if not graph.get(curr):
        return
    for v in graph[curr]:
        #一个顶点在当前递归路径中只能出现一次，否则会陷入死循环。
        if v in path:
            print("v %s in path %s" %(v, path))
            continue
        #构造下次递归的父路径
        path.append(v)
        find_path_all(v,end,path)
        path.pop()

path_all = []
find_path_all('A', 'G',path=['A'])
print(path_all)
```

``` py
#从图中找出任意一条从起始顶点到终止顶点的路径
def find_path(graph, start, end, path=[]):
    if start == end:
        print "path", path
        return True
    if not graph.get(start):
        path.pop()
        return False
    for v in graph[start]:
        if v not in path:
            path.append(v)
            if find_path(graph,v,end,path):
                return True
    return False

path = []
if find_path(graph, 'A', 'C', path=path):
    print(path)
else:
    print(1)
```

## 二叉搜索树中第K小的元素

中序遍历是一个排好序的数组，可以用迭代加上count来中断，加速判断，也可以保存这个数组，方便每一次取第k个最小数

``` py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        if not root:
            return None
        s=[]
        node=root
        res=[]
        while node or s:
            while node:
                s.append(node)
                node=node.left
            node=s.pop()
            res.append(node.val)
            node=node.right
        return res[k-1]
```

## 课程表II

BFS，入度为0进队列

``` py
import collections
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj = collections.defaultdict(list)
        indegree=[0 for _ in range(numCourses)]
        for cur,pre in prerequisites:
            adj[pre].append(cur)
            indegree[cur]+=1
        res=[]
        queue=collections.deque()
        for i in range(numCourses):
            if indegree[i]==0:
                queue.append(i)
        while queue:
            cur=queue.popleft()
            res.append(cur)
            numCourses-=1
            for nb in adj[cur]:
                indegree[nb]-=1
                if indegree[nb]==0:
                    queue.append(nb)
        if numCourses==0:
            return res
        else:
            return []
```

DFS，标记-1, 1, 0，用flag来表示

``` py
import collections

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj = collections.defaultdict(list)
        for cur, pre in prerequisites:
            adj[pre].append(cur)
        flag = [0 for _ in range(numCourses)]
        res = []

        def dfs(i, adj, flag):
            # 如果没有环，就返回True
            if flag[i] == -1:  # 被别的线程访问过，直接返回
                return True
            if flag[i] == 1:  # 当前线程访问过，有环
                return False
            flag[i] = 1
            for nb in adj[i]:
                if dfs(nb, adj, flag) == False:
                    return False
            flag[i] = -1
            res.append(i)
            return True
        for i in range(numCourses):
            if dfs(i, adj, flag) == False:
                return []
        return res[::-1]
```

## 存在重复元素

``` py
import collections
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dic=collections.Counter(nums)
        for k in dic:
            if dic[k]>=2:
                return True
        return False
```

## 数组中的第K个最大元素

``` py
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        q=heapq.nlargest(k,nums)
        return q[-1]
```

## 天际线问题

首先看一下优先级队列，其实python有heapq可以直接在list里实现最小堆。

``` py
q=[]
heapq.heappush(q, (3, 'code'))
heapq.heappush(q, (2, 'eat'))
heapq.heappush(q, (5, 'fuck'))
a=heapq.heappop(q)
print(a)
```

扫描法

## 单词搜索 II

DFS方法超出时间限制

``` py
from typing import List
import heapq

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        row = len(board)
        col = len(board[0])

        def dfs(i, j, flag, word):
            if word == '':
                return True
            directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            for k in range(4):
                cur_i = i+directions[k][0]
                cur_j = j+directions[k][1]
                if cur_i < 0 or cur_i >= row or cur_j < 0 or cur_j >= col or flag[cur_i][cur_j] == 1:
                    continue
                else:
                    if board[cur_i][cur_j] == word[0]:
                        flag[cur_i][cur_j] = 1
                        if dfs(cur_i, cur_j, flag, word[1:]):
                            flag[cur_i][cur_j] = 0
                            return True
                        else:
                            flag[cur_i][cur_j] = 0
        if not board:
            return []
        dic = {}
        for k in words:
            if k[0] in dic:
                dic[k[0]].append(k)
            else:
                dic[k[0]] = [k]
        print(dic)
        res = []
        flag = [[0 for _ in range(col)] for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if board[i][j] in dic:
                    for word in dic[board[i][j]]:
                        flag[i][j] = 1
                        if dfs(i, j, flag, word[1:]) == True:
                            res.append(word)
                            dic[board[i][j]].remove(word)
                        flag[i][j] = 0
        return res

func = Solution()
res = func.findWords(
    [["o", "a", "a", "n"], ["e", "t", "a", "e"], [
        "i", "h", "k", "r"], ["i", "f", "l", "v"]],
    ["oath", "pea", "eat", "rain"])
# print(res)
```

## 二叉树的公共节点

``` txt
当我们用递归去做这个题时不要被题目误导，应该要明确一点
这个函数的功能有三个：给定两个节点 pp 和 qq

如果 pp 和 qq 都存在，则返回它们的公共祖先；
如果只存在一个，则返回存在的一个；
如果 pp 和 qq 都不存在，则返回NULL
本题说给定的两个节点都存在，那自然还是能用上面的函数来解决

具体思路：
（1） 如果当前结点 rootroot 等于NULL，则直接返回NULL
（2） 如果 rootroot 等于 pp 或者 qq ，那这棵树一定返回 pp 或者 qq
（3） 然后递归左右子树，因为是递归，使用函数后可认为左右子树已经算出结果，用 leftleft 和 rightright 表示
（4） 此时若leftleft为空，那最终结果只要看 rightright；若 rightright 为空，那最终结果只要看 leftleft
（5） 如果 leftleft 和 rightright 都非空，因为只给了 pp 和 qq 两个结点，都非空，说明一边一个，因此 rootroot 是他们的最近公共祖先
（6） 如果 leftleft 和 rightright 都为空，则返回空（其实已经包含在前面的情况中了）

时间复杂度是O(n)O(n)：每个结点最多遍历一次或用主定理，空间复杂度是O(n)O(n)：需要系统栈空间
```

``` py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root==None:
            return None
        if root==p or root==q:
            return root
        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)
        if left==None:
            return right
        if right==None:
            return left
        if left and right:
            return root
        return None
```

## 除自身以外数组的乘积

``` py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        n=len(nums)
        pre=[0 for _ in range(n)]
        mul=1
        for i in range(n):
            pre[i]=mul
            mul*=nums[i]
        las=[0 for _ in range(n)]
        mul=1
        for i in range(n-1,-1,-1):
            las[i]=mul
            mul*=nums[i]
        res=[0 for _ in range(n)]
        for i in range(n):
            res[i]=pre[i]*las[i]
        return res
```

## 滑动窗口最大值

heapq的删除元素方法

``` py
# OlogN
h[i] = h[-1]
h.pop()
heapq._siftup(h, i)
heapq._siftdown(h, 0, i)
```

``` py
# ON
h[i] = h[-1]
h.pop()
heapq.heapify(h)
```

用了最小堆来写，复杂度应该是在O(NK)

``` py
import heapq
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        q=[]
        res=[]
        nums=[-w for w in nums]
        for i in range(k):
            heapq.heappush(q,nums[i])
        res.append(q[0]*-1)
        for i in range(k,len(nums)):
            q.remove(nums[i-k])
            heapq.heappush(q,nums[i])
            heapq.heapify(q)
            res.append(q[0]*-1)
        return res
```

可以用单调队列来进一步降低复杂度O(N)

``` py
# 暂时还没有想出来
```

## 搜索二维矩阵 II

剑指offer上面一毛一样的题目

``` py
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # 这不剑指offer里面的题目嘛，从右上方开始搜素，往左下方搜索
        if not matrix:
            return False
        n=len(matrix)
        m=len(matrix[0])
        i=0
        j=m-1
        while 0<=i<=n-1 and 0<=j<=m-1:
            if matrix[i][j]==target:
                return True
            elif target<matrix[i][j]:
                j-=1
            elif matrix[i][j]<target:
                i+=1
        return False
```

## 有效的字母异位词

哈希表

``` py
import collections
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dic1=collections.Counter(s)
        for k in t:
            if k not in dic1:
                return False
            else:
                dic1[k]-=1
            if dic1[k]==0:
                dic1.pop(k)
        if dic1=={}:
            return True
        else:
            return False
```

## 生命游戏

一种需要另外开辟空间的方法

``` py
class Solution:
    def getlife(self,board,i,j):
        n=len(board)
        m=len(board[0])
        directions=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
        count=0
        for direct in directions:
            cur_i=i+direct[0]
            cur_j=j+direct[1]
            if 0<=cur_i<=n-1 and 0<=cur_j<=m-1:
                if board[cur_i][cur_j]==1:
                    count+=1
        return count

    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None
        n=len(board)
        m=len(board[0])
        mark=[[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                count=self.getlife(board,i,j)
                if board[i][j]==1:
                    if count<2:
                        mark[i][j]=0
                    elif 2<=count<=3:
                        mark[i][j]=1
                    elif count>3:
                        mark[i][j]=0
                else:
                    if count==3:
                        mark[i][j]=1
        for i in range(n):
            for j in range(m):
                board[i][j]=mark[i][j]
```

一种不需要额外空间的办法，可以用奇偶性来判断是否是活细胞或者死细胞

## 