# 剑指offer

## 二维数组的查找 P44

从右上角开始寻找，不断减少col和增加row

``` python
class Solution:
    def search_num(self, nums, number):
        if nums == []:
            return False
        n = len(nums)
        m = len(nums[0])
        row = 0
        col = m-1
        while 0 <= row < n and 0 <= col < m:
            if nums[row][col] == number:
                return [row, col]
            if nums[row][col] > number:
                col -= 1
            else:
                row += 1
        return False

nums = [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
func = Solution()
print(func.search_num(nums=nums, number=1))
```

## 链表或者树

记住链表的创建

``` python
class TreeNode:
        def __init__(self, x):
            self.val = x
            self.next = None
```

## 根据前序和中序遍历结果输出整个树

前序:[1, 2, 4, 7, 3, 5, 6, 8]

中序:[4, 7, 2, 1, 5, 3, 8, 6]

思路：前序找到root，然后查找root在中序中的位置，root左边的是左子树序列，右边的右子树序列，再用递归的方法来写。

完整的代码，由于之后可能会手撕代码，还是全部一起写完比较好。对python里的类别，类别函数，初始化更加了解。

``` python
class Node:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Tree_order:
    def __init__(self, root):
        self.pre_list = []
        self.mid_list = []
        self.preorder(root)
        self.midorder(root)

    def preorder(self, node):
        if not node:
            return
        self.pre_list.append(node.val)
        self.preorder(node.left)
        self.preorder(node.right)

    def midorder(self, node):
        if not node:
            return
        self.midorder(node.left)
        self.mid_list.append(node.val)
        self.midorder(node.right)

class Solution:
    def reconstruct(self, pre, mid):
        if pre == []:
            return None
        if len(pre)!=len(mid):
            return False
        root = Node(pre[0])
        if len(pre) == 1:
            return root
        for i in range(len(mid)):
            if mid[i] == pre[0]:
                break
        else:
            return False
        pre_l = pre[1:i+1]
        pre_r = pre[i+1:]

        mid_l = mid[:i]
        mid_r = mid[i+1:]

        root.left = self.reconstruct(pre_l, mid_l)
        root.right = self.reconstruct(pre_r, mid_r)

        return root

if __name__ == "__main__":
    pre = [1, 2, 4, 7, 3, 5, 6, 8]
    mid = [4, 7, 2, 1, 5, 3, 8, 6]
    func = Solution()
    tr = func.reconstruct(pre, mid)
    Tr_func = Tree_order(tr)
    print(Tr_func.pre_list)
    print(Tr_func.mid_list)

```

### 考虑特殊的情况

这一点很重要，能判断到底你能不能写出超级鲁邦的代码应对各种情况。有以下几点

* 普通正确的二叉树
* 只有反方向节点的二叉树，只有一个节点的二叉树
* 输入空的list比如[]，输入pre和mid不是正确对应关系的list

想法，设置一个全局的变量来控制递归过程中会出现的错误，最后的输出需要考虑flag的值。

``` python
class Node:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Tree_order:
    def __init__(self, root):
        self.pre_list = []
        self.mid_list = []
        self.preorder(root)
        self.midorder(root)

    def preorder(self, node):
        if not node:
            return
        self.pre_list.append(node.val)
        self.preorder(node.left)
        self.preorder(node.right)

    def midorder(self, node):
        if not node:
            return
        self.midorder(node.left)
        self.mid_list.append(node.val)
        self.midorder(node.right)

class Solution:
    def __init__(self):
        self.flag = True

    def reconstruct(self, pre, mid):
        if pre == []:
            return None
        root = Node(pre[0])
        if len(pre) != len(mid):
            self.flag = False
        if len(pre) == 1:
            if pre[0] != mid[0]:
                self.flag = 1
            else:
                return root
        for i in range(len(mid)):
            if mid[i] == pre[0]:
                break
        else:
            self.flag = False
        pre_l = pre[1:i+1]
        pre_r = pre[i+1:]

        mid_l = mid[:i]
        mid_r = mid[i+1:]

        root.left = self.reconstruct(pre_l, mid_l)
        root.right = self.reconstruct(pre_r, mid_r)
        if self.flag:
            return root
        else:
            return None

if __name__ == "__main__":
    pre = [1, 2, 4, 7, 3, 5, 6, 10]
    mid = [4, 7, 2, 1, 5, 3, 8, 6]
    # pre = [1, 2, 4]
    # mid = [2, 1, 4]

    func = Solution()
    tr = func.reconstruct(pre, mid)
    Tr_func = Tree_order(tr)
    print(Tr_func.pre_list)
    print(Tr_func.mid_list)
```

## 二叉树的下一个节点

考虑几种情况

* 有右子树的时候，从右子树的最左边的节点为后一个节点
* 没有右子树的时候，需要看自己是父节点的左子树还是右子树，如果自己是左子树，那么直接输出父节点即可。
* 如果没有右子树，而且自己还是父节点的右子树，沿着父节点一直往上找，直到找到一个节点是他自己父节点的左节点，那么这个父节点就是输出的节点，或者直接找到了根节点也行。

``` python
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return
        #如果该节点有右子树，那么下一个节点就是它右子树中的最左节点
        elif pNode.right!=None:
            pNode=pNode.right
            while pNode.left!=None:
                pNode=pNode.left
            return pNode
        #如果一个节点没有右子树，，并且它还是它父节点的右子节点
        elif pNode.next!=None and pNode.next.right==pNode:
            while pNode.next!=None and pNode.next.left!=pNode:
                pNode=pNode.next
            return pNode.next
        #如果一个节点是它父节点的左子节点，那么直接返回它的父节点
        else:
            return pNode.next
```

## 两个栈实现队列

``` python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.s1=[]
        self.s2=[]
    def push(self, node):
        # write code here
        self.s1.append(node)
    def pop(self):
        # return xx
        if self.s2==[]:
            while self.s1:
                self.s2.append(self.s1.pop())
            return self.s2.pop()
        else:
            return self.s2.pop()
```

## 旋转数组的最小数字

二分查找的变体，需要考虑的东西挺多的，比如输入的数是没有旋转的。输入的数是10111这种前面后面一样的只能通过顺序查找了。

``` python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if rotateArray == []:
            return 0
        l = 0
        r = len(rotateArray)-1
        if rotateArray[l] < rotateArray[r]:
            return rotateArray[0]
        if rotateArray[l] == rotateArray[r]:
            min_num = rotateArray[l]
            for i in range(len(rotateArray)):
                if rotateArray[i] < min_num:
                    min_num = rotateArray[i]
            return min_num
        while l < r:
            # print(l, r)
            if r-l == 1:
                return rotateArray[r]
            else:
                mid = l+(r-l)//2
                if rotateArray[l] <= rotateArray[mid]:
                    l = mid
                elif rotateArray[mid] <= rotateArray[r]:
                    r = mid

func = Solution()
nums = [9, 10, 1, 2, 3, 4, 5, 6, 7, 8]
res = func.minNumberInRotateArray(nums)
print(res)

```

## 斐波那契数列

DP方法！！！

``` python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n==0:
            return 0
        if n==1:
            return 1
        if n==2:
            return 1
        pre_1=1
        pre_2=1
        for i in range(3,n+1):
            cur=pre_1+pre_2
            pre_1,pre_2=pre_2,cur
        return pre_2
```

## 跳台阶

DP法

``` python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        n=number
        if n==0:
            return 0
        if n==1:
            return 1
        if n==2:
            return 2
        pre_1=1
        pre_2=2
        for i in range(3,n+1):
            cur=pre_1+pre_2
            pre_1,pre_2=pre_2,cur
        return pre_2
```

## 变态跳台阶

贪心算法，其实还是DP，但是这个可以通过数学计算出来一个公式比较牛逼。

要想跳到第n级台阶，就可以从第n-1级、第n-2级、***、第1级 跳到第n级，再加上直接从地面到第n级的一种情况。

f(n)=f(n-1)+f(n-2)+ ... +f(1)+1

同理f(n-1)=f(n-2)+f(n-3)+ ... +f(1)+1

减一下能够得到公式f(n)=2*f(n-1)，所以其实是一个数学问题，f(n)=2^(n-1)

``` python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        return 2**(number-1)
```

## 矩形覆盖

``` python
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
        n=number
        if n==0:
            return 0
        if n==1:
            return 1
        if n==2:
            return 2
        pre1=1
        pre2=2
        for i in range(3,n+1):
            cur=pre1+pre2
            pre1,pre2=pre2,cur
        return pre2
```

## 二进制中1的个数

* 考虑n是正数还是负数的情况
* n和n-1进行and判断可以讲n中最后一个1和之后的值全部变成0，这样只要通过几次and操作就可以把整个数变成0.

``` python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        count = 0
        while n:
            count += 1
            n = (n-1) & n
        return count
```

## 数值的整数次方

思路，考虑一些东西

* 整数是否是正数还是负数
* 数值是否是0

``` python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0 and exponent <= 0:
            return None
        if exponent == 0:
            return 1
        if exponent > 0:
            sum = 1
            for i in range(exponent):
                sum *= base
            return sum
        if exponent < 0:
            exponent *= -1
            sum = 1
            for i in range(exponent):
                sum *= base
            return 1.0/sum
```

## 调整数组顺序使奇数位于偶数前面

思路，要保证调整之后奇偶数各自内部的顺序是不变的

一个想法，两个list，不断取值然后判断，最后两个list连起来

python可以用sorted和lambda函数来解决，设置顺序为是否能被2整除

``` python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        return sorted(array,key=lambda c:c%2==0)
```

### 扩展的功能

* 按照大小分为两部分，负数排在非负数前面
* 能被3整除的放在不能被3整除的前面

其实就是修改lambda函数，改一下key的结构

下面考虑先按照负数和非负数排，然后再按照能不能被3整除排的代码

``` python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        return sorted(array, key=lambda c: (c > 0, c % 3 != 0))
```

## 链表中倒数第k个节点

考虑：大家都知道用双指针，但是要考察的主要内容其实是鲁棒性，程序会不会崩溃才是面试官考察的东西，所以慢点写考虑全面比较好。

错误：

* 空链表
* 长度比k小
* k是一个无意义的数，比如0或者负数

``` python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if not head:
            return None
        if k<=0:
            return None
        left_node=head
        right_node=head
        for i in range(k-1):
            if not right_node.next:
                return None
            right_node=right_node.next
        while right_node.next:
            left_node=left_node.next
            right_node=right_node.next
        return left_node
```

## 反转链表

需要注意的点：

* 边界条件判断一下，比如空节点，单个节点什么的
* 头结点并不是反转之后的尾巴，而是需要再加上一个None节点来当尾巴！！！

``` python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead:
            return None
        if pHead.next==None:
            return pHead
        pre=None ## 很重要！！！
        cur=pHead
        las=cur.next
        while cur.next:
            cur.next=pre
            pre=cur
            cur=las
            las=cur.next
        cur.next=pre
        return cur
```

## 合并两个排序的链表

考虑的问题

* 怎么合并事先要想好
* 特殊情况怎么处理，空节点什么的

我的想法就是把2插到1里面

``` python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        if pHead1.val > pHead2.val:
            pHead1, pHead2 = pHead2, pHead1
        cur1 = pHead1
        cur2 = pHead2
        while cur1.next and cur2:
            if cur1.next.val >= cur2.val:
                cur1.next, cur1.next.next, cur2 = cur2, cur1.next, cur2.next
            else:
                cur1 = cur1.next
        if not cur1.next:
            cur1.next = cur2
        return pHead1
```

## 树的子结构

需要注意的一点就是中间有一个判断过程，遍历所有节点的时候可能会出现找到一个节点之后进入一个死循环，不会接着判断下一个条件。所以这里不能直接用return，而是用分别判断左右子树来判断。

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSubtree(self,node1,node2):
        if not node2:
            return True
        if not node1:
            return False
        if node1.val==node2.val:
            return self.isSubtree(node1.left,node2.left) and self.isSubtree(node1.right,node2.right)
        else:
            return False

    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        res=False
        if pRoot1 and pRoot2:
            if pRoot1.val==pRoot2.val:
                # 这里不能直接return
                res=self.isSubtree(pRoot1,pRoot2)
            if res==False:
                res=self.HasSubtree(pRoot1.left,pRoot2) or self.HasSubtree(pRoot1.right,pRoot2)
        return res
```

## 二叉树的镜像

递归遍历所有节点，前序。在遍历的过程中交换节点的左右节点顺序。

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return None
        if root.left or root.right:
            root.left,root.right=root.right,root.left
            if root.left:
                self.Mirror(root.left)
            if root.right:
                self.Mirror(root.right)
        return root
```

## 顺时针打印矩阵

需要考虑中间的循环，单独写成一个函数，注意边界条件宁愿多设置一些if语句，防止出错。

``` python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def list_matrix(self,row1,col1,row2,col2,matrix):
        res=[]
        if row1!=row2 and col1!=col2:
            for c in range(col1,col2):
                res.append(matrix[row1][c])
            for r in range(row1,row2):
                res.append(matrix[r][col2])
            for c in range(col2,col1,-1):
                res.append(matrix[row2][c])
            for r in range(row2,row1,-1):
                res.append(matrix[r][col1])
        if row1==row2 and col1==col2:
            res.append(matrix[row1][col1])
        if row1==row2 and col1!=col2:
            for c in range(col1,col2+1):
                res.append(matrix[row1][c])
        if row1!=row2 and col1==col2:
            for r in range(row1,row2+1):
                res.append(matrix[r][col1])
        return res
    def printMatrix(self, matrix):
        # write code here
        if matrix==[]:
            return None
        n=len(matrix)
        m=len(matrix[0])
        row1=0
        col1=0
        row2=n-1
        col2=m-1
        ans=[]
        while row1<=row2 and col1<=col2:
            ans.extend(self.list_matrix(row1,col1,row2,col2,matrix))
            row1+=1
            col1+=1
            row2-=1
            col2-=1
        return ans
```

## 包含min函数的栈

难点在于要想到使用辅助栈，不能固定思维，在O1复杂度的情况下必须要用额外的空间和额外的辅助变量参数才行，所以想到给每一个数一个标记，但是这个标记会随着数被pop出去消失，所以想到要用同步辅助栈，直接压入当前最小值。

``` python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.res = []
        self.helper = []

    def push(self, node):
        # write code here
        self.res.append(node)
        if self.helper == []:
            self.helper.append(node)
        else:
            cur = self.helper.pop()
            if cur <= node:
                self.helper.append(cur)
                self.helper.append(cur)
            else:
                self.helper.append(cur)
                self.helper.append(node)

    def pop(self):
        # write code here
        if self.res == []:
            return None
        else:
            cur = self.res.pop()
            self.helper.pop()
        return cur

    def top(self):
        # write code here
        cur = self.res.pop()
        self.res.append(cur)
        return cur

    def min(self):
        # write code here
        cur = self.helper.pop()
        self.helper.append(cur)
        return cur
```

## 栈的压入弹出序列

要考虑的一点是中间有一个判断，我用的是index来判断现在用到了pushV中的哪一个，所以在while之前就需要来一个是否越界的条件，然后在while中也需要一个是否越界的条件判断。

``` python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if pushV == [] and popV == []:
            return True
        if len(pushV) != len(popV):
            return False
        index = 0
        s = []
        for k in popV:
            if s == [] or s[-1] != k:
                if index == len(pushV):
                    return False
                while pushV[index] != k:
                    s.append(pushV[index])
                    index += 1
                    if index == len(pushV):
                        return False
                index += 1
            else:
                s.pop()
        return True
```

## 从上往下打印二叉树

BFS，广度优先遍历二叉树，队列

deque通常的用法

``` python
d = collections.deque([])
d.append('a') # 在最右边添加一个元素，此时 d=deque('a')
d.appendleft('b') # 在最左边添加一个元素，此时 d=deque(['b', 'a'])
d.extend(['c','d']) # 在最右边添加所有元素，此时 d=deque(['b', 'a', 'c', 'd'])
d.extendleft(['e','f']) # 在最左边添加所有元素，此时 d=deque(['f', 'e', 'b', 'a', 'c', 'd'])
d.pop() # 将最右边的元素取出，返回 'd'，此时 d=deque(['f', 'e', 'b', 'a', 'c'])
d.popleft() # 将最左边的元素取出，返回 'f'，此时 d=deque(['e', 'b', 'a', 'c'])
d.rotate(-2) # 向左旋转两个位置（正数则向右旋转），此时 d=deque(['a', 'c', 'e', 'b'])
d.count('a') # 队列中'a'的个数，返回 1
d.remove('c') # 从队列中将'c'删除，此时 d=deque(['a', 'e', 'b'])
d.reverse() # 将队列倒序，此时 d=deque(['b', 'e', 'a'])
```

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
import collections
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        q=collections.deque()
        q.append(root)
        res=[]
        while q:
            cur=q.popleft()
            res.append(cur.val)
            if cur.left:
                q.append(cur.left)
            if cur.right:
                q.append(cur.right)
        return res
```

## 二叉搜索树的后序遍历序列

思路：对任意一序列，都是先找根节点，然后左子树的子序列，直到第一个比根节点大的值，然后当做右子树的子序列，如果没有左子树或者右子树，那么返回True，如果出现了第一个比根节点大的值之后又出现了一个比根节点小的值，那么这个序列肯定不是正确的。

醉了，空的list系统判断不是一个正确的后序遍历序列，虽然考虑了，但是面试的时候一定要问一下面试官，空的序列到底算不算正确的，空的后序遍历序列对应的难道不是空的二叉搜索树，空的树就不是树了？？？

``` python
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if sequence==[]:
            return False
        n=len(sequence)
        if n==1:
            return True
        root_val=sequence[-1]
        right_st=n-1
        for i in range(n-1):
            if sequence[i]<root_val and right_st!=n-1:
                return False
            if sequence[i]>root_val and right_st==n-1:
                right_st=i
        if right_st==0 or right_st==n-1:
            return self.VerifySquenceOfBST(sequence[:-1])
        else:
            return self.VerifySquenceOfBST(sequence[:right_st+1]) and self.VerifySquenceOfBST(sequence[right_st:-1])

```

## 二叉树中和为某一值的路

考虑一下中序遍历，其实就是DFS遍历

``` python
class Solution:
    def levelOrderBottom(self, root):
        """
        中序遍历 非递归
        :param root:  根节点
        :return: list_node -> List
        """
        s = []
        node = root
        list_node = []
        while s or node:
            while node:
                s.append(node)
                node = node.left
            node = s.pop()
            print(node.val)
            list_node.append(node.val)
            node = node.right
        return list_node
```

复习一下后序

``` python
class Solution:
    def levelOrderBottom(self, root):
        """
        后序遍历 非递归
        :param root:  根节点
        :return: list_node -> List
        """
        if not root:
            return []
        s=[root]
        list_node=[]
        while s:
            node=s.pop()
            if node.left:
                s.append(node.left)
            if node.right:
                s.append(node.right)
            list_node.append(node.val)
        list_node=list_node[::-1]
        return list_node
```

在stack的操作过程中一定要注意是不是copy了list，还是直接引用了list的指针！！！

这个在s.append((node, list))的时候一定要非常注意，被坑了不知道多少次！！！

改成s.append((node, list[:])才行，是需要copy这个list才行的

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
        s=[]
        res=[]
        node=root
        tmp=[]
        while s or node:
            while node:
                if node==root:
                    tmp.append(node.val)
                else:
                    tmp.append(tmp[-1]+node.val)
                s.append((node,tmp[:]))
                node=node.left
            node,tmp=s.pop()
            print(node.val,tmp)
            if not node.left and not node.right and tmp[-1]==expectNumber:
                res.append(tmp)
            node=node.right
        for lis in res:
            for i in range(len(lis)-1,0,-1):
                lis[i]=lis[i]-lis[i-1]
        return res
```

## 复杂链表的复制

思路就是用一个dic来保存所有的N和对应的复制节点N'，这样第一次遍历先创建next的所有链接，第二次遍历创建random的所有链接.

``` python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if not pHead:
            return None
        dic={}
        copy_head=RandomListNode(pHead.label)
        cur_copy=copy_head
        dic[pHead]=copy_head
        pre_copy=cur_copy
        cur_ori=pHead.next
        while cur_ori:
            cur_copy=RandomListNode(cur_ori.label)
            dic[cur_ori]=cur_copy
            pre_copy.next=cur_copy
            pre_copy=cur_copy
            cur_ori=cur_ori.next
        cur_ori=pHead
        cur_copy=copy_head
        while cur_ori:
            if cur_ori.random:
                cur_copy.random=dic[cur_ori.random]
            cur_ori=cur_ori.next
            cur_copy=cur_copy.next
        return copy_head
```

## 二叉搜索树和双向链表

一开始的思路是递归计算，通过40%的案例，出问题的地方，案例是只有右节点的树。比如{1, #, 2, #, 3, #, 4, #, 5}

对应输出应该为: From left to right are:1, 2, 3, 4, 5; From right to left are:5, 4, 3, 2, 1

你的输出为: From left to right are:1, 2, 3, 4, 5; From right to left are:5

也就是说我的节点都没left节点？

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def connect(self, node):
        if not node:
            return None
        if not node.left and node.right:
            return node
        if node.left:
            l=self.connect(node.left)
            while l.right:
                l=l.right
            node.left=l
            l.right=node
        if node.right:
            r=self.connect(node.right)
            while r.left:
                r=r.left
            node.right=r
            r.left=node
        return node
    def Convert(self, pRootOfTree):
        # write code here
        midnode=self.connect(pRootOfTree)
        while midnode.left:
            midnode=midnode.left
        return midnode
```

那么咋改？肯定是出现了节点判断的问题，或者是左右子树判断出错，找了一下，居然是最sb的地方写错了。当然还有一个问题是需要判断一开始的节点到底是不是空的，返回None。

``` python
if not node.left and not node.right:
    return node
```

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def connect(self, node):
        if not node:
            return None
        if not node.left and not node.right:
            return node
        if node.left:
            l=self.connect(node.left)
            while l.right:
                l=l.right
            node.left=l
            l.right=node
        if node.right:
            r=self.connect(node.right)
            while r.left:
                r=r.left
            node.right=r
            r.left=node
        return node
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return None
        midnode=self.connect(pRootOfTree)
        while midnode.left:
            midnode=midnode.left
        return midnode
```

## 字符串的排列

有重复数字，全排列，肯定DP，虽然书上是递归，不太清楚到底哪一个更好

首先创立一个dic用来存k个字符的全排列list，在加入一个字符之后取出dic[k]中的每一个word，进行ch的插入操作，形成一个新的字符串，去掉重复的。但是最后需要来一个sort按字典序来排列。整个算法非常直观，可能会有一些操作多加了一些复杂度。

``` python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.dic = {}
        self.dic[0] = []

    def sort_ch(self, lis, ch):
        k = len(lis)
        if k == 0:
            self.dic[k+1] = [ch]
        else:
            pre = self.dic[k]
            self.dic[k+1] = []
            for word in pre:
                for j in range(len(word)+1):
                    add_word = word[:j]+ch+word[j:]
                    if add_word not in self.dic[k+1]:
                        self.dic[k+1].append(add_word)

    def Permutation(self, ss):
        # write code here
        if ss == '':
            return []
        n = len(ss)
        for i in range(n):
            print(ss[:i], ss[i])
            self.sort_ch(ss[:i], ss[i])
        res = self.dic[n]
        res.sort()
        return res
```

## 数组中出现次数超过一半的数字

直接一个dic或者collections. Counter(numbers)解决问题。

``` python
# -*- coding:utf-8 -*-
import collections
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        dic=collections.Counter(numbers)
        n=len(numbers)
        for k in dic:
            if dic[k]>n//2:
                return k
        return 0
```

## 最小的K个数

商谈二面的题目，醉了, 当时复杂度太高凉了。

思路：创建一个大小为K个容器来存储k个数字，然后在n个数字中遍历，读一个数再加入到这个k组中。所以其实是需要用二叉树来进行比较，插入，删除这三个操作，时间复杂度为O(logN)，有点像二分查找。那么就要用最大堆或者红黑树来解决这个问题。刚好我全部不会。

``` python
def quicksort(num, low, high):  # 快速排序
    if low < high:
        location = partition(num, low, high)
        quicksort(num, low, location - 1)
        quicksort(num, location + 1, high)

def partition(data_list,begin,end):
    #选择最后一个元素作为分区键
    partition_key = data_list[end]

    #index为分区键的最终位置
    #比partition_key小的放左边，比partition_key大的放右边
    index = begin
    for i in range(begin,end):
        if data_list[i] < partition_key:
            data_list[i],data_list[index] = data_list[index],data_list[i]
            index+=1

    data_list[index],data_list[end] = data_list[end],data_list[index]
    return index

def findkth(num, low, high, k):  # 找到数组里第k个数
    index = partition(num, low, high)
    if index == k:
        return num[index]
    if index < k:
        return findkth(num, index+1, high, k)
    else:
        return findkth(num, low, index-1, k)

pai = [2, 3, 1, 5, 4, 6]
# quicksort(pai, 0, len(pai) - 1)
print(findkth(pai, 0, len(pai)-1, 3))
```

用class再写一遍

``` python
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k <= 0 or k > len(tinput):
            return []

        def partition(num, low, high):
            key_num = num[high]
            index = low
            for i in range(low, high):
                if num[i] < key_num:
                    num[index], num[i] = num[i], num[index]
                    index += 1
            num[index], num[high] = num[high], num[index]
            return index

        def select_min_k(num, low, high, index_k):
            index = partition(num, low, high)
            if index == index_k:
                return index
            else:
                if index > index_k:
                    return select_min_k(num, low, index-1, index_k)
                if index < index_k:
                    return select_min_k(num, index+1, high, index_k)

        res = select_min_k(tinput, 0, len(tinput)-1, k-1)
        print(tinput)
        return tinput[:res+1]
```

### 快速排序

对于一串序列，首先从中选取一个数，凡是小于这个数的值就被放在左边一摞，凡是大于这个数的值就被放在右边一摞。然后，继续对左右两摞进行快速排序。递归的方法

直到进行快速排序的序列长度小于 2 （即序列中只有一个值或者空值）。

``` python
class Solution:
    def quicksort(self, seq):
        if len(seq) < 2:
            return seq
        else:
            base = seq[0]
            left = [elem for elem in seq[1:] if elem < base]
            right = [elem for elem in seq[1:] if elem > base]
            return self.quicksort(left)+[base]+self.quicksort(right)
```

``` python
class Solution:
    def quicksort2(self, a, left, right):
            if left > right:
                return
            t = a[left]
            i = left
            j = right

            while i != j:
                while a[j] >= t and i < j:
                    j -= 1
                while a[i] < t and i < j:
                    i += 1
                a[i], a[j] = a[j], a[i]
            a[left], a[i] = a[i], a[left]
            self.quicksort2(a, left, i-1)
            self.quicksort2(a, i+1, right)
            return a
```

### 用快速排序思想首先的O(N)时间复杂度的算法

``` python
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        n = len(tinput)
        if k <= 0 or k > n:
            return []
        start = 0
        end = n-1
        mid = self.partition(tinput, start, end)
        while mid != k-1:
            if k-1 > mid:
                start = mid+1
                mid = self.partition(tinput, start, end)
            elif k-1 < mid:
                end = mid-1
                mid = self.partition(tinput, start, end)
        res = tinput[:k]
        res.sort()
        return res

    def partition(self, numbers, low, right):
        st = low
        l = low
        r = right
        key = numbers[l]
        while l < r:
            while l < r and numbers[r] >= key:
                r -= 1
            while l < r and numbers[l] <= key:
                l += 1
            numbers[l], numbers[r] = numbers[r], numbers[l]
        numbers[l], numbers[st] = numbers[st], numbers[l]
        return l
```

### 最大最小堆的实现

首先要看一下python是怎么创建堆的，需要用到一个heapq库。

一行代码版, 醉了

``` python
# -*- coding:utf-8 -*-
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        return [] if k<=0 or k>len(tinput) else heapq.nsmallest(k, tinput)
```

复杂一点的代码, heap只有最小堆，那么怎么实现最大堆呢，很简单，把元素取反，然后最小的元素其实就是没有改变之前的最大元素。

用K容器的代码，维护一个最小堆。堆的维护插入的时间复杂度为O(logK), 遍历全部的数字一次为O(N，所以最后的时间复杂度为O(NlogK)，特别适合在N很大，K不是很大的时候用这个方法，不会太占用额外的空间。

``` python
import heapq

class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        n = len(tinput)
        if k <= 0 or k > n:
            return []
        heap_small = []
        for i in range(k):
            heapq.heappush(heap_small, tinput[i]*(-1))
        for i in range(k, n):
            if tinput[i]*(-1) > heap_small[0]:
                heapq.heapreplace(heap_small, tinput[i]*(-1))
        return [k*(-1) for k in heap_small]
```

## 连续子数组的最大和

DP方法，以index结尾的子数组最大和计算一下，往后dp

``` python 

# -*- coding:utf-8 -*-

class Solution:

    def FindGreatestSumOfSubArray(self, array):
        # write code here
        dic = {}
        max_num = float("-inf")
        for i in range(len(array)):
            if i == 0:
                dic[i] = array[i]
            else:
                if dic[i-1] < 0:
                    dic[i] = array[i]
                else:
                    dic[i] = dic[i-1]+array[i]
            if dic[i] > max_num:
                max_num = dic[i]
        return max_num

``` 

## 整数中1出现的次数

笨办法，直接判断个位数是不是1，num%10==1，然后整除//10

``` python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        res=0
        for num in range(1,n+1):
            while num!=0:
                if num%10==1:
                    res+=1
                num=num//10
        return res
```

## 把数组排成最小的数

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

分析：

``` python
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ''
        numbers=[str(num) for num in numbers]
        for i in range(1,len(numbers)):
            for j in range(len(numbers)-i):
                if numbers[j]+numbers[j+1]>numbers[j+1]+numbers[j]:
                    numbers[j],numbers[j+1]=numbers[j+1],numbers[j]
        res=''
        for word in numbers:
            res+=word
        return res
```

## 丑数

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

``` python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index<=0:
            return 0
        count=1
        result=1
        result_lis=set()
        while count<index:
            result_lis.add(2*result)
            result_lis.add(3*result)
            result_lis.add(5*result)
            result=min(result_lis)
            result_lis.remove(result)
            count+=1
        return result
```

## 第一次只出现一次的字符

用dic换时间

``` python
# -*- coding:utf-8 -*-
import collections
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        dic={}
        dic_id={}
        s=list(s)
        for i in range(len(s)):
            if s[i] not in dic:
                dic[s[i]]=[i,1]
            else:
                dic[s[i]][1]+=1
        if dic:
            min_id=len(s)
            for k in dic:
                if dic[k][1]==1 and dic[k][0]<min_id:
                    min_id=dic[k][0]
            return min_id
        else:
            return -1
```

## 数组中的逆序对

首先需要看一下归并排序到底是什么

``` python
class Solution:
    def __init__(self):
        self.count = 0

    def merge(self, left, right):
        tmp = []
        p1 = 0
        p2 = 0
        while p1 <= len(left)-1 and p2 <= len(right)-1:
            if left[p1] < right[p2]:
                tmp.append(left[p1])
                p1 += 1
            else:
                tmp.append(right[p2])
                p2 += 1
        while p1 <= len(left)-1:
            tmp.append(left[p1])
            p1 += 1
        while p2 <= len(right)-1:
            tmp.append(right[p2])
            p2 += 1
        return tmp

    def merge_sort(self, arr):
        if len(arr) == 0:
            return []
        if len(arr) == 1:
            return arr
        mid = len(arr)//2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self.merge(left, right)

    def InversePairs(self, data):
        # write code here
        if len(data) < 2:
            return 0
        res = self.merge_sort(data)
        return res

func = Solution()
print(func.InversePairs(data=[1, 5, 6, 7, 2, 3, 4, 0]))
```

注意在python里面用self函数修改list的值也可以，但是不好理解，不如直接return一个新的数组来代替组合什么的。

递归需要注意先写终止条件，再写递归，最后输出是什么

有了以上几条就可以开始写逆序对了

* 分析一下，逆序对求数量的代码应该是写在self.merge里面，合并left和right的时候需要比较两个list中的数值，我们可以从简单到复杂来看，比如left和right中只有一个数字的时候会怎么样。
* 如果left比right中的数字大，那么就count+=1然后排序组合好以后就不再管内部他们自己的顺序了，已经排好序
* 现在来看有多个数字的情况，left[i]比right[j]小，tmp.append(left[i])，left[i]与right[j]后面均不可能构成逆序对，i+=1之后继续
* 如果left[i]>right[j]了，那么left[i]和right[j]可以构成一个逆序对，并且left[i]后面的数都能和right[j]构成逆序对，那么tmp.appnend(right[j])，j+=1之后继续
* 这样肯定会有一个list最先走到头，那么直接把剩下来的那个接到tmp后面即可，中间的一些逆序对已经计算过了个数

根据上述分析得到代码如下，其实只要改一处地方即可

``` python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.count = 0

    def merge(self, left, right):
        tmp = []
        p1 = 0
        p2 = 0
        while p1 <= len(left)-1 and p2 <= len(right)-1:
            if left[p1] < right[p2]:
                tmp.append(left[p1])
                p1 += 1
            else:
                # 就是改的这里
                self.count += len(left)-1-p1+1
                tmp.append(right[p2])
                p2 += 1
        while p1 <= len(left)-1:
            tmp.append(left[p1])
            p1 += 1
        while p2 <= len(right)-1:
            tmp.append(right[p2])
            p2 += 1
        return tmp

    def merge_sort(self, arr):
        if len(arr) == 0:
            return []
        if len(arr) == 1:
            return arr
        mid = len(arr)//2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self.merge(left, right)

    def InversePairs(self, data):
        # write code here
        if len(data) < 2:
            return 0
        res = self.merge_sort(data)
        return self.count%1000000007
```

## 两个链表的第一个公共节点

一般看到这种题从后往前看比较好，从最后的一个节点开始往前遍历得到最后一个相同的节点即可，那么就有点像先进后出，用栈来表示

``` python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        s_1=[]
        s_2=[]
        if not pHead1 or not pHead2:
            return None
        node=pHead1
        while node:
            s_1.append(node)
            node=node.next
        node=pHead2
        while node:
            s_2.append(node)
            node=node.next
        pre=None
        while s_1 and s_2:
            cur_1=s_1.pop()
            cur_2=s_2.pop()
            if cur_1!=cur_2:
                return pre
            else:
                pre=cur_1
        return pre
```

## 数字在排序数组中出现的次数

2行，直接用collections库

``` python
# -*- coding:utf-8 -*-
import collections
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        dic=collections.Counter(data)
        return dic[k]
```

要是不用collections。Counter也能做，重点是分析复杂度

需要对二分查找非常熟悉才行

``` python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if data == []:
            return 0
        l = self.getfirstnum(data, k)
        r = self.getlastnum(data, k)
        return r-l+1

    def getfirstnum(self, data, k):
        l = 0
        r = len(data)-1
        while l <= r:
            mid = (l+r)//2
            if data[mid] < k:
                l = mid+1
            else:
                r = mid-1
        return l

    def getlastnum(self, data, k):
        l = 0
        r = len(data)-1
        while l <= r:
            mid = (l+r)//2
            if data[mid] <= k:
                l = mid+1
            else:
                r = mid-1
        return r
```

## 二叉树的深度

DFS没什么说的，注意我是用的栈实现的，用迭代的算法

``` python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        s=[]
        node=pRoot
        high=0
        max_depth=0
        while node or s:
            while node:
                high+=1
                s.append((node,high))
                node=node.left
            node,high=s.pop()
            if not node.left and not node.right:
                if high>max_depth:
                    max_depth=high
            node=node.right
        return max_depth
```

## 二叉搜索树的第K大节点

需要用到中序遍历，全局的变量

``` python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.count=0
        self.res=None

    def mid_order(self,node):
        if node.right:
            self.mid_order(node.right)
        if self.count==1:
            self.res=node
        self.count-=1
        if node.left:
            self.mid_order(node.left)

    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root or k<=0:
            return None
        self.count=k
        self.mid_order(root)
        return self.res.val
```

## 平衡二叉树

递归算法，判断左右子树是不是平衡二叉树，平衡二叉树表示的是左右子树的高度差小于等于1的二叉树，使用到了一个求数的高度的函数，这样不是最简单的，最简单的是使用后序遍历的算法最简单。

``` python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def getdepth(self,root):
        if not root:
            return 0
        left=self.getdepth(root.left)
        right=self.getdepth(root.right)
        return max(left,right)+1
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:
            return True
        left=self.getdepth(root.left)
        right=self.getdepth(root.right)
        if abs(left-right)>1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
```

## 数组中数字出现的次数

要求比较严格，时间是O(n), 空间是O(1)，所以所有排序全部不能用，因为排序的复杂度至少为O(NlogN)

另外找方法，使用位运算，异或属性，这里需要看一下异或是什么

异或的性质，两个数字异或的结果a^b是将 a 和 b 的二进制每一位进行运算，得出的数字。 运算的逻辑是，如果同一位的数字相同则为 0，不同则为 1

异或的规律，任何数和本身异或则为0，任何数和 0 异或是本身

异或满足交换律，比如17^19^17=19，这个非常重要

### 1个数字出现一次，其他数字出现两次的算法

全员异或的结果就是出现一次的数，需要应用到交换律和数字与0异或结果为数字

``` python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        single_number = 0
        for num in nums:
            single_number=single_number^num
        return single_number
```

### 2个数字出现一次，其他出现两次

我们进行一次全员异或操作，得到的结果就是那两个只出现一次的不同的数字的异或结果。

我们刚才讲了异或的规律中有一个任何数和本身异或则为0， 因此我们的思路是能不能将这两个不同的数字分成两组 A 和 B。

分组需要满足两个条件.

两个独特的的数字分成不同组

相同的数字分成相同组

这样每一组的数据进行异或即可得到那两个数字。

``` python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        ret = 0  # 所有数字异或的结果
        a = 0
        b = 0
        for n in nums:
            ret ^= n
        # 找到第一位不是0的
        h = 1
        while(ret & h == 0):
            h <<= 1
        for n in nums:
            # 根据该位是否为0将其分为两组
            if (h & n == 0):
                a ^= n
            else:
                b ^= n

        return [a, b]
```

## 1个数字出现一次，其他出现3次

还是用位运算的特点，如果一个数出现过了3次，那么该二进制位数上如果有1的地方应该1的总和是能被3整除的，如果这个位置的1的和不能被3整除，那么出现一次的那个数字在这个位置一定是1

``` python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        if not nums: 
            return []
        res=0# 最终计算出来的数，即要找出的数字
        #计算32位上每一位的和，如果这个位上的和，不能被3整除，这要找个这个数字上的bit位是1
        for i in range(32):
            bit_sum=0#每一位上的和
            mask=1<<i#每一次要操作的位
            for num in nums:
                if num&mask !=0:
                    bit_sum+=1
            if bit_sum%3!=0:
                res=res|mask# 如果此位上为1，每一次都和mask或运算
        return res
```

## 和为s的两个数字

我的想法是dic存起来，时间O(n)，每个遍历一次

``` python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic={}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [nums[i],target-nums[i]]
            else:
                dic[target-nums[i]]=nums[i]
        return None
```

书上说可以用双指针来做，应该也可以，很简单写一下

``` python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if len(nums)<2:
            return None
        left=0
        right=len(nums)-1
        while left<right:
            if nums[left]+nums[right]==target:
                return [nums[left],nums[right]]
            if nums[left]+nums[right]<target:
                left+=1
            if nums[left]+nums[right]>target:
                right-=1
        return None
```

## 和为s的连续正数序列

数学题，就是一个一元二次方程

``` python
import math
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        res=[]
        for x in range(1,target//2+1):
            c=x-x*x-2*target
            y=(math.sqrt(1-4*c)-1)/2
            if y==int(y):
                y=int(y)
                res.append([i for i in range(x,y+1)])
        return res

```

## 翻转单词顺序

用好stripe(), split()

``` python
class Solution:
    def reverseWords(self, s: str) -> str:
        s=s.strip()
        lis=s.split()[::-1]
        return ' '.join(lis)
```

## 左旋转字符串

``` python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:]+s[:n]
```

## 滑动窗口最大值

这里用一个暴力法加减少运算的一个方法，有更好的方法，队列什么的，这里就不用了

``` python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k<=0 or k>len(nums):
            return []
        res=[]
        lis=nums[0:k]
        max_temp=max(lis)
        res.append(max_temp)
        for i in range(k,len(nums)):
            lis=nums[i-k+1:i+k]
            if nums[i]>max_temp:
                max_temp=nums[i]
            elif max_temp==nums[i-k]:
                max_temp=max(nums[i-k+1:i+1])
            res.append(max_temp)
        return res
```

##　队列里的最大值

果然使用了双边队列，一个队列保存最大值，一个队列保存值，和求栈里的最大值异曲同工。

``` python
import collections
class MaxQueue:
    def __init__(self):
        self.helper=collections.deque()
        self.q=collections.deque()

    def max_value(self) -> int:
        if self.helper:
            return self.helper[0]
        else:
            return -1

    def push_back(self, value: int) -> None:
        self.q.append(value)
        while self.helper and self.helper[-1]<value:
            self.helper.pop()
        self.helper.append(value)

    def pop_front(self) -> int:
        if self.q:
            if self.helper[0]==self.q[0]:
                self.helper.popleft()
                return self.q.popleft()
            else:
                return self.q.popleft()
        else:
            return -1

# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```

## n个骰子的点数

DP方法，已知n-1个骰子的点数分布情况，求出再加一个骰子的点数分布情况

dp[n][j] += dp[n-1][j - i]

考虑好边界情况，我们可以直接知道的状态是啥，就是第一阶段的状态：投掷完 11 枚骰子后，它的可能点数分别为 1, 2, 3, ... , 6并且每个点数出现的次数都是1

这个需要动一下脑子，有点绕晕了，虽然最后还是写出来了，但是中间的list变量意义，索引意义都要搞明白

``` python 
class Solution:

    def twoSum(self, n: int):
        num_sum = [0 for i in range(n+1)]
        dp = [[] for i in range(n+1)]
        dp[0].append(1)
        for j in range(1, 6+1):
            dp[1].append(1)
            num_sum[1] = 6
        for i in range(2, n+1):
            for j in range(i, 6*i+1):
                count = 0
                for k in range(1, 6+1):
                    if i-1 <= j-k <= 6*(i-1):
                        count += dp[i-1][j-k-(i-1)]
                dp[i].append(count)
            num_sum[i] = num_sum[i-1]*6
        return [x/num_sum[n] for x in dp[n]]

``` 

## 扑克牌中的顺子

从[0,0,1,2,...,13]抽取5个数字，查看是不是顺子，其中0可以当做任意的数字！！！

函数的输入就是一个list，包含5个数字，从上面得到的，函数用来判断True or False

这道题的中间判断还是有点多的，必须要考虑一些情况

```python
class Solution:
    def isStraight(self, nums):
        if len(nums) < 5:
            return False
        nums.sort()
        # print(nums)
        l = 0
        r = 4
        for i in range(0, 5):
            if nums[i] != 0:
                l = i
                break
        dic = {}
        for i in range(l, r+1):
            if nums[i] in dic:
                return False
            else:
                dic[nums[i]] = 1
        if nums[r]-nums[l] <= 4:
            return True
        else:
            return False

```

## 圆圈中最后剩下的数字

约瑟夫环问题，链表来模拟, 没用，超时，很坑，不过算的是对的

``` python
class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        if n<=0:
            return None
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
```

公式法，其实就是数学运算

f(N, M)=(f(N−1, M)+M)%N

理解这个递推式的核心在于关注胜利者的下标位置是怎么变的。每杀掉一个人，其实就是把这个数组向前移动了M位。然后逆过来，就可以得到这个递推式

现在改为人数改为N，报到M时，把那个人杀掉，那么数组是怎么移动的？

每杀掉一个人，下一个人成为头，相当于把数组向前移动M位。若已知N-1个人时，胜利者的下标位置位f(N−1, M)f(N-1, M)f(N−1, M)，则N个人的时候，就是往后移动Ms

``` python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        p=0
        for i in range(2,n+1):
            p=(p+m)%i
        return p
```

## 股票的最大利润

时间O(N),类似于Dp

``` python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # dp问题吧，dp[i]为i天之前的最小值
        if not prices:
            return 0
        dp=[0]*len(prices)
        dp[0]=prices[0]
        max_price=0
        for i in range(1,len(prices)):
            dp[i]=min(dp[i-1],prices[i])
            max_price=max(max_price,prices[i]-dp[i])
        return max_price
```

## 