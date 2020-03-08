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

## 寻找两个有序数组的中位数

题目中需要的是时间复杂度在O(log(min(m, n)))，这种时间复杂度明显是需要二分查找或者递归的算法，而且min(m, n)提示了

中位数可以首先在第一个数组里面找一个中间的数，比如第i个数，那么在nums2里面有j个数，如果需要划分一刀把两个数字分开的话就一定要左右个数一样，或者左边比右边多一个数。

为了简化代码，不分情况讨论，我们使用一个小trick，我们分别找第 (m+n+1) / 2 个，和 (m+n+2) / 2 个，然后求其平均值即可，这对奇偶数均适用。加入 m+n 为奇数的话，那么其实 (m+n+1) / 2 和 (m+n+2) / 2 的值相等，相当于两个相同的数字相加再除以2，还是其本身。

``` python

```

