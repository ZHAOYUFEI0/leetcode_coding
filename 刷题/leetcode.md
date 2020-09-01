### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

#### 思路：

##### 排序题

##### 思路一：$nlog(n)$排序

##### 思路二：$klog(n)$堆排序

#### 库函数

##### 手写堆排

##### 思路三：平均时间$O(n)$，最差$O(n^2)$部分的<u>快排</u>

在数组中，随机选择一个值pivot，将数组分成两个部分，left和right,然后再比较len(right)与k-1的大小，如果len(right)==k-1,则返回pivot为第K大的值，其余当len(right)大于或小于k-1,分别递归findKthLargest(right, k)和findKthLargest(left, k-len(right)-1)即可。

##### 代码:

##### 思路一：排序

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return sorted(nums, reverse = True)[k - 1]
```

##### 思路二：堆排序

1， 库函数

nlargest(n, irerable, key) 返回n个最大值

nsmallest(n, irerable, key) 返回n个最小值

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
      #返回n个最大值
        return heapq.nlargest(k, nums)[-1]

```

最小堆

```python
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        l = []
        for i in nums[:k]:
            heapq.heappush(l,i)
        for i in nums[k:]:
            if i > l[0]:
                heapq.heappop(l)
                heapq.heappush(l,i)
        return l[0]
```



##### 思路二：

2， 手写

```python
class Solution:
  def findKthLargest(self, nums: List[int], k: int) -> int:
    def adjust_heap(idx, max_len):
        left = 2 * idx + 1
        right = 2 * idx + 2
        max_loc = idx
        if left < max_len and nums[max_loc] < nums[left]:
            max_loc = left
        if right < max_len and nums[max_loc] < nums[right]:
            max_loc = right
        if max_loc != idx:
            nums[idx], nums[max_loc] = nums[max_loc], nums[idx]
            adjust_heap(max_loc, max_len)
    
    # 建堆
    n = len(nums)
    for i in range(n // 2 - 1, -1, -1):
        adjust_heap(i, n)
    #print(nums)
    res = None
    for i in range(1, k + 1):
        #print(nums)
        res = nums[0]
        nums[0], nums[-i] = nums[-i], nums[0]
        adjust_heap(0, n - i)
    return res
```
##### 思路三：快排

```python
from random import randint
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        index = randint(0,len(nums)-1)
        left = [i for i in nums[:index]+nums[index+1:] if i < nums[index]]
        right = [i for i in nums[:index]+nums[index+1:] if i >= nums[index]]
        if len(right) == k-1:
            return nums[index]
        elif len(right) > k-1:
            return self.findKthLargest(right,k)
        else:
            return self.findKthLargest(left, k-len(right)-1)
```



#### [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)



给定一个字符串 `s` ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 `s` 的最大长度为 `1000` 。

##### 解题思路：

首先明确一下 base case，如果只有一个字符，显然最长回文子序列长度是 1，也就是 dp[i][j] = 1 (i == j)。

因为 i 肯定小于等于 j，所以对于那些 i > j 的位置，根本不存在什么子序列，应该初始化为 0。

另外，看看刚才写的状态转移方程，想求 dp\[i][j]需要知道 dp\[i+1][j-1]，dp\[i+1][j]，dp\[i][j-1] 这三个位置；再看看我们确定的 base case，填入 dp 数组之后是这样：

<img src="/Users/zhaoyufei/leetcode_coding/刷题/最长回文子序列.jpg" alt="最长回文子序列" style="zoom:50%;" />

**为了保证每次计算 `dp[i][j]`，左下右方向的位置已经被计算出来，只能斜着遍历或者反着遍历**：

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s) 
        dp = [[0] * n for _ in range(n)]
        for j in range(0, n):
            dp[j][j] = 1
            for i in range(j-1, -1, -1):
                if s[i] == s[j]: dp[i][j] = dp[i+1][j-1] + 2
                else : dp[i][j] = max(dp[i+1][j] , dp[i][j-1])
        return dp[0][n-1]


```

