# -*- coding:utf-8 -*-
import heapq


# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ''
        numbers = [str(num) for num in numbers]
        for i in range(1, len(numbers)):
            for j in range(len(numbers)-i):
                if numbers[j]+numbers[j+1] > numbers[j+1]+numbers[j]:
                    numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
        res = ''
        for word in numbers:
            res += word
        return res


func = Solution()
print(func.PrintMinNumber(numbers=[3, 32, 321]))

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
