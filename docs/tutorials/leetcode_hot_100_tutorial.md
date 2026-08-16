# LeetCode HOT 100：一天速记清单（Python）

> 目标不是一天重新做完 100 题，而是：看到题名后，能在 10 秒内说出算法、关键不变量和复杂度；重点题能默写骨架。
> 
> 标记：⭐⭐⭐ 必须默写；⭐⭐ 能独立讲清并补出关键代码；⭐ 只需恢复思路。官方题单：[LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/)
> 
> 

## 一天怎么用

- **08:30–09:00**：读“15 个母模板”，遮住代码口述不变量。

- **09:00–11:00**：哈希、双指针、滑窗、子串、数组、矩阵（21 题）。

- **11:10–12:20**：链表（14 题）。重点手写反转、快慢指针、归并、LRU。

- **13:30–15:10**：二叉树、图（19 题）。重点手写 DFS 返回值、BFS、拓扑排序。

- **15:20–16:20**：回溯、二分（14 题）。统一套模板，不逐题死背。

- **16:30–17:20**：栈、堆、贪心（12 题）。重点单调栈、双堆、区间边界。

- **18:30–20:10**：动态规划、多维 DP（15 题）。每题只说清状态、转移、初始化、遍历顺序。

- **20:20–20:50**：技巧（5 题）。

- **21:00–22:00**：只默写所有 ⭐⭐⭐；卡住超过 3 分钟立刻看答案并重写。

- **22:00–22:20**：100 题闪回验收：每题 10 秒，只说“识别信号 → 算法 → 关键坑”。

## 15 个母模板（优先背）

### 1\. 哈希：边遍历边查询 ⭐⭐⭐

```python
seen = {}
for i, x in enumerate(nums):
    if target - x in seen:
        return [seen[target - x], i]
    seen[x] = i
```

### 2\. 可变滑动窗口 ⭐⭐⭐

```python
left = 0
for right, x in enumerate(s):
    # 把 x 纳入窗口
    while 窗口不合法:
        # 移出 s[left]
        left += 1
    ans = max(ans, right - left + 1)
```

最小覆盖类：`right` 扩到满足，`left` 尽量缩；无重复类：一旦冲突就缩到合法。

### 3\. 前缀和 \+ 哈希 ⭐⭐⭐

```python
cnt = {0: 1}
pre = ans = 0
for x in nums:
    pre += x
    ans += cnt.get(pre - k, 0)
    cnt[pre] = cnt.get(pre, 0) + 1
```

### 4\. 单调栈 ⭐⭐⭐

```python
st = []                         # 下标；保持对应值单调
for i, x in enumerate(nums):
    while st and nums[st[-1]] < x:
        j = st.pop()
        ans[j] = i - j
    st.append(i)
```

### 5\. 单调队列（窗口最大值）⭐⭐⭐

```python
from collections import deque
q = deque()                     # 下标；值递减
for i, x in enumerate(nums):
    while q and nums[q[-1]] <= x: q.pop()
    q.append(i)
    if q[0] <= i - k: q.popleft()
    if i >= k - 1: ans.append(nums[q[0]])
```

### 6\. 二分：左边界 ⭐⭐⭐

```python
l, r = 0, len(nums)             # [l, r)
while l < r:
    m = (l + r) // 2
    if nums[m] >= target: r = m
    else: l = m + 1
# l 是第一个 >= target 的位置
```

### 7\. 链表反转 \+ 虚拟头 ⭐⭐⭐

```python
prev, cur = None, head
while cur:
    nxt = cur.next
    cur.next = prev
    prev, cur = cur, nxt
return prev

dummy = ListNode(0, head)        # 删除/拼接头节点时先想 dummy
```

### 8\. 链表快慢指针 ⭐⭐⭐

```python
slow = fast = head
while fast and fast.next:
    slow, fast = slow.next, fast.next.next
```

判环后求入口：相遇时另设 `p=head`，`p` 与 `slow` 同速走，再相遇即入口。

### 9\. 二叉树 DFS：先定义返回值 ⭐⭐⭐

```python
def dfs(node):
    if not node: return 空树值
    left = dfs(node.left)
    right = dfs(node.right)
    # 用 left/right 更新全局答案
    return 当前子树要交给父节点的值
```

### 10\. BFS / 拓扑排序 ⭐⭐⭐

```python
from collections import deque
q = deque(所有起点)
while q:
    for _ in range(len(q)):
        x = q.popleft()
        for y in neighbors(x):
            if 未访问或入度降为0: q.append(y)
```

### 11\. 回溯 ⭐⭐⭐

```python
def dfs(start, path):
    if 满足答案条件:
        ans.append(path[:])
        return
    for i in range(start, len(nums)):
        if 需要剪枝: continue
        path.append(nums[i])
        dfs(下一状态, path)
        path.pop()
```

排列用 `used`；组合/子集用 `start`；棋盘搜索额外做“标记—递归—恢复”。

### 12\. 一维 DP / 0\-1 背包 ⭐⭐⭐

```python
dp = [False] * (target + 1)
dp[0] = True
for x in nums:
    for j in range(target, x - 1, -1):  # 倒序：每个 x 只用一次
        dp[j] |= dp[j - x]
```

完全背包若物品可重复，容量通常正序。

### 13\. 区间 DP / 二维 DP ⭐⭐

```python
for length in range(2, n + 1):
    for l in range(n - length + 1):
        r = l + length - 1
        dp[l][r] = ...
```

### 14\. 堆：维护 Top K ⭐⭐

```python
import heapq
h = []
for x in stream:
    heapq.heappush(h, x)
    if len(h) > k: heapq.heappop(h)
```

### 15\. 贪心：遍历时维护“当前最优可达边界” ⭐⭐

```python
far = 0
for i, step in enumerate(nums):
    if i > far: return False
    far = max(far, i + step)
return True
```

---

## 100 题闪回清单

### 一、哈希（3）

#### 1\. 两数之和｜思路详解与标准代码

* [ ] ⭐⭐⭐ **1\. 两数之和**：边扫边查 `target-x`；必须先查后存，避免复用自己。$O(n)$。

**思路：**看到“两个数凑目标且要返回下标”，优先边遍历边查询哈希表。哈希表始终只保存当前元素之前的值到下标；先查 target\-x，再存当前值，命中即返回。

**易错点：**

- 必须先查后存，避免同一元素复用

- 重复值应保留可形成答案的历史下标

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, value in enumerate(nums):
            complement = target - value
            if complement in seen:
                return [seen[complement], i]
            seen[value] = i
        return []
```

#### 49\. 字母异位词分组｜思路详解与标准代码

* [ ] ⭐⭐ **49\. 字母异位词分组**：用排序串或 26 维计数元组作 key；相同 key 入同组。

**思路：**看到“按字母组成分组”，应构造与排列无关的规范键。将每个字符串排序后的结果作为键，把原串追加到同一列表。

**易错点：**

- 键必须可哈希

- 空字符串也应正确归组

**复杂度：**时间 O\(n·k log k\)，额外空间 O\(nk\)，k 为字符串最大长度

```python
from typing import List, Optional
from collections import Counter, defaultdict, deque
from bisect import bisect_left
from math import inf, sqrt
from itertools import pairwise
from functools import cache, reduce
from operator import xor
import heapq

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            k = ''.join(sorted(s))
            d[k].append(s)
        return list(d.values())
```

#### 128\. 最长连续序列｜思路详解与标准代码

* [ ] ⭐⭐⭐ **128\. 最长连续序列**：放入集合，只从 `x-1` 不存在的序列起点向右数；别从每个数重复扩展。$O(n)$ 期望。

**思路：**看到“无序数组中的连续整数长度”，用集合获得常数期望查询。只从 x\-1 不在集合的序列起点向右扩展，保证每个数至多被有效扫描一次。

**易错点：**

- 先去重

- 只从序列起点扩展

- 空数组返回 0

**复杂度：**期望时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List, Optional
from collections import Counter, defaultdict, deque
from bisect import bisect_left
from math import inf, sqrt
from itertools import pairwise
from functools import cache, reduce
from operator import xor
import heapq

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        ans = 0
        d = defaultdict(int)
        for x in nums:
            y = x
            while y in s:
                s.remove(y)
                y += 1
            d[x] = d[y] + y - x
            ans = max(ans, d[x])
        return ans
```

### 二、双指针（4）

#### 283\. 移动零｜思路详解与标准代码

* [ ] ⭐⭐ **283\. 移动零**：`slow` 指向下一个非零落点，`fast` 扫描并交换；保持相对顺序。

**思路：**看到“原地移动且保持非零相对顺序”，使用快慢指针稳定压缩。fast 扫描非零元素，slow 指向下一个落点，交换后推进 slow。

**易错点：**

- 必须原地修改

- 保持非零元素相对顺序

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List, Optional
from collections import Counter, defaultdict, deque
from bisect import bisect_left
from math import inf, sqrt
from itertools import pairwise
from functools import cache, reduce
from operator import xor
import heapq

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        k = 0
        for i, x in enumerate(nums):
            if x:
                nums[k], nums[i] = nums[i], nums[k]
                k += 1
```

#### 11\. 盛最多水的容器｜思路详解与标准代码

* [ ] ⭐⭐⭐ **11\. 盛最多水的容器**：面积由短板决定；计算后移动短板，移动长板不可能变优。

**思路：**看到面积由两端距离和短板决定，从最宽区间开始双指针收缩。每轮更新面积后只移动较短的一端，因为移动长板无法突破当前短板上限。

**易错点：**

- 先计算再移动

- 相等时移动任一端都正确

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            t = min(height[l], height[r]) * (r - l)
            ans = max(ans, t)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans
```

#### 15\. 三数之和｜思路详解与标准代码

* [ ] ⭐⭐⭐ **15\. 三数之和**：排序；固定 `i`，左右夹逼；`i/l/r` 都要去重，和大右移、和小左移。

**思路：**看到三元组求和且答案需去重，先排序再固定一个数做双指针。固定值、左值和右值都按层去重，并依据当前和移动边界。

**易错点：**

- 外层和内层都要去重

- 排序后 nums\[i\]\>0 可提前结束

- 返回值不能含重复三元组

**复杂度：**时间 O\(n²\)，除结果外额外空间 O\(log n\)（排序栈）

```python
from typing import List, Optional
from collections import Counter, defaultdict, deque
from bisect import bisect_left
from math import inf, sqrt
from itertools import pairwise
from functools import cache, reduce
from operator import xor
import heapq

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n - 2):
            if nums[i] > 0:
                break
            if i and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, n - 1
            while j < k:
                x = nums[i] + nums[j] + nums[k]
                if x < 0:
                    j += 1
                elif x > 0:
                    k -= 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j, k = j + 1, k - 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans
```

#### 42\. 接雨水｜思路详解与标准代码

* [ ] ⭐⭐⭐ **42\. 接雨水**：左右最大值双指针；较小一侧的上界已确定，累加 `边界-当前高度`。

**思路：**看到每列积水取决于左右最高柱，使用双指针维护 left\_max 与 right\_max。当左侧上界较小时左列水量已经确定，否则处理右列，随后向内收缩。

**易错点：**

- 先更新对应侧最大值再计算水量

- 空数组和短数组自然返回 0

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        water = 0
        while left <= right:
            if left_max <= right_max:
                left_max = max(left_max, height[left])
                water += left_max - height[left]
                left += 1
            else:
                right_max = max(right_max, height[right])
                water += right_max - height[right]
                right -= 1
        return water
```

### 三、滑动窗口（2）

#### 3\. 无重复字符的最长子串｜思路详解与标准代码

* [ ] ⭐⭐⭐ **3\. 无重复字符的最长子串**：右扩，重复时左缩；或记录字符最后位置并令 `left=max(left,last+1)`。

**思路：**看到“最长且窗口内无重复”，用字符最后出现位置维护可行窗口。右端扫描字符，left 只能向右跳到 last\[ch\]\+1，窗口始终无重复。

**易错点：**

- left 必须取 max，不能回退

- 答案用 right\-left\+1

**复杂度：**时间 O\(n\)，额外空间 O\(min\(n, 字符集大小\)\)

```python
from typing import List, Optional
from collections import Counter, defaultdict, deque
from bisect import bisect_left
from math import inf, sqrt
from itertools import pairwise
from functools import cache, reduce
from operator import xor
import heapq

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        cnt = Counter()
        ans = l = 0
        for r, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] > 1:
                cnt[s[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```

#### 438\. 找到字符串中所有字母异位词｜思路详解与标准代码

* [ ] ⭐⭐ **438\. 找到字符串中所有字母异位词**：固定长度窗口维护 26 计数/差值；窗口长度等于 `p` 时判相等。

**思路：**看到“固定模式串的所有排列起点”，使用长度固定为 len\(p\) 的滑动窗口。右侧加入、超长时移出左侧，窗口计数与目标计数一致就记录起点。

**易错点：**

- p 比 s 长时返回空

- 记录的是窗口左端下标

**复杂度：**时间 O\(n\)，额外空间 O\(字符集大小\)

```python
from typing import List
from collections import Counter

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m, n = len(s), len(p)
        ans = []
        if m < n:
            return ans
        cnt1 = Counter(p)
        cnt2 = Counter(s[: n - 1])
        for i in range(n - 1, m):
            cnt2[s[i]] += 1
            if cnt1 == cnt2:
                ans.append(i - n + 1)
            cnt2[s[i - n + 1]] -= 1
        return ans
```

### 四、子串（3）

#### 560\. 和为 K 的子数组｜思路详解与标准代码

* [ ] ⭐⭐⭐ **560\. 和为 K 的子数组**：前缀和；此前每个 `pre-k` 都形成一个答案，故哈希存“次数”而非位置。

**思路：**看到“连续子数组和等于 k”且元素可为负数，使用前缀和频次哈希。扫描到前缀和 pre 时，所有历史 pre\-k 都能形成答案，再把 pre 计数加入哈希。

**易错点：**

- 初始化 count\[0\]=1

- 哈希存频次而非单个位置

- 先统计答案再加入当前前缀和

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List
from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        count[0] = 1
        prefix = 0
        answer = 0
        for value in nums:
            prefix += value
            answer += count[prefix - k]
            count[prefix] += 1
        return answer
```

#### 239\. 滑动窗口最大值｜思路详解与标准代码

* [ ] ⭐⭐⭐ **239\. 滑动窗口最大值**：递减双端队列存下标；队首过期弹，队尾不大于新值弹。

**思路：**看到固定窗口反复求最大值，用存下标的单调递减双端队列。每个右端到来时先移除过期队首，再弹出不大于新值的队尾，窗口形成后读取队首。

**易错点：**

- 队列必须存下标以判断过期

- 相等值可弹旧留新

- 仅在 i\>=k\-1 时输出

**复杂度：**时间 O\(n\)，额外空间 O\(k\)

```python
from typing import List
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = deque()
        answer = []
        for i, value in enumerate(nums):
            while queue and queue[0] <= i - k:
                queue.popleft()
            while queue and nums[queue[-1]] <= value:
                queue.pop()
            queue.append(i)
            if i >= k - 1:
                answer.append(nums[queue[0]])
        return answer
```

#### 76\. 最小覆盖子串｜思路详解与标准代码

* [ ] ⭐⭐⭐ **76\. 最小覆盖子串**：计数窗口；满足全部种类后不断缩左并更新最短，缩坏后再扩右。

**思路：**看到“包含目标全部字符且允许重复”的最短子串，使用可变滑动窗口和需求计数。右端扩张使缺失总数归零后，左端尽量收缩并更新最短答案；移出必需字符后恢复扩张。

**易错点：**

- 按字符总数而非字符种类判断满足

- 更新答案必须在缩坏之前

- 无解返回空串

**复杂度：**时间 O\(len\(s\)\+len\(t\)\)，额外空间 O\(字符集大小\)

```python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s:
            return ""
        need = Counter(t)
        missing = len(t)
        left = 0
        best_start, best_len = 0, float('inf')
        for right, ch in enumerate(s):
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1
            while missing == 0:
                window_len = right - left + 1
                if window_len < best_len:
                    best_start, best_len = left, window_len
                left_ch = s[left]
                need[left_ch] += 1
                if need[left_ch] > 0:
                    missing += 1
                left += 1
        if best_len == float('inf'):
            return ""
        return s[best_start:best_start + best_len]
```

### 五、普通数组（5）

#### 53\. 最大子数组和｜思路详解与标准代码

* [ ] ⭐⭐⭐ **53\. 最大子数组和**：`cur=max(x,cur+x)`，`ans=max(ans,cur)`；负贡献就从当前重启。

**思路：**看到连续子数组最大和，使用 Kadane 状态。cur 表示必须以当前位置结尾的最大和，取“单独开始”与“接在前面”较大者，并同步更新全局答案。

**易错点：**

- 答案不能初始化为 0，否则全负数组错误

- 空输入不在题目约束内

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = f = nums[0]
        for x in nums[1:]:
            f = max(f, 0) + x
            ans = max(ans, f)
        return ans
```

#### 56\. 合并区间｜思路详解与标准代码

* [ ] ⭐⭐⭐ **56\. 合并区间**：按左端点排序；新区间左端点 `> 当前右端点` 才断开，否则扩右端点。

**思路：**看到重叠区间合并，先按左端点排序。结果末区间与新区间不相交时追加，否则只扩展末区间右端点。

**易错点：**

- 端点相接属于重叠

- 右端点取 max，不能直接覆盖

**复杂度：**时间 O\(n log n\)，额外空间 O\(n\)（含输出）

```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        ans = []
        st, ed = intervals[0]
        for s, e in intervals[1:]:
            if ed < s:
                ans.append([st, ed])
                st, ed = s, e
            else:
                ed = max(ed, e)
        ans.append([st, ed])
        return ans
```

#### 189\. 轮转数组｜思路详解与标准代码

* [ ] ⭐⭐ **189\. 轮转数组**：`k%=n`；整体反转，再反转前 `k` 与后 `n-k`。

**思路：**看到原地右轮转，利用三次反转把后 k 个元素搬到前面。先令 k 对 n 取模，再整体反转并分别反转前后两段。

**易错点：**

- 必须先 k%=n

- 题目保证 n\>0

- 原地修改无返回值

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        def reverse(i: int, j: int):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i, j = i + 1, j - 1

        n = len(nums)
        k %= n
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
```

#### 238\. 除自身以外数组的乘积｜思路详解与标准代码

* [ ] ⭐⭐⭐ **238\. 除自身以外数组的乘积**：答案先存左乘积，再用一个变量从右累乘；不用除法，$O(1)$ 额外空间。

**思路：**看到不能除法且要求线性时间，用前后缀乘积。答案先写入每个位置左侧乘积，再从右向左用一个变量乘上右侧乘积。

**易错点：**

- 正确处理一个或多个 0

- 额外空间不计输出数组

**复杂度：**时间 O\(n\)，除输出外额外空间 O\(1\)

```python
from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        left = right = 1
        for i, x in enumerate(nums):
            ans[i] = left
            left *= x
        for i in range(n - 1, -1, -1):
            ans[i] *= right
            right *= nums[i]
        return ans
```

#### 41\. 缺失的第一个正数｜思路详解与标准代码

* [ ] ⭐⭐⭐ **41\. 缺失的第一个正数**：原地把 `x∈[1,n]` 放到 `nums[x-1]`；最后首个错位下标 `i` 答 `i+1`。

**思路：**看到“最小缺失正数、线性时间、常数空间”，把数组当作下标哈希。通过循环交换把值 x 放到 nums\[x\-1\]；最后第一个 nums\[i\]\!=i\+1 的位置就是答案。

**易错点：**

- 只处理 1\<=x\<=n

- 交换前判断目标位置值不同以防死循环

- 重复值和负数不能作为下标

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                target = nums[i] - 1
                nums[i], nums[target] = nums[target], nums[i]
        for i, value in enumerate(nums):
            if value != i + 1:
                return i + 1
        return n + 1
```

### 六、矩阵（4）

#### 73\. 矩阵置零｜思路详解与标准代码

* [ ] ⭐⭐ **73\. 矩阵置零**：首行首列充当标记，另存首列是否应清零；先标记、再清内部、最后清边界。

**思路：**看到原地把含零行列清零，用首行首列保存标记。先单独记录首列是否含零，再标记内部，按标记清内部，最后处理首行首列。

**易错点：**

- 标记阶段不能提前清零

- 首行和首列状态要分开保存

**复杂度：**时间 O\(mn\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        if not matrix or not matrix[0]:
            return
        rows, cols = len(matrix), len(matrix[0])
        first_col_zero = any(matrix[row][0] == 0 for row in range(rows))
        for row in range(rows):
            for col in range(1, cols):
                if matrix[row][col] == 0:
                    matrix[row][0] = 0
                    matrix[0][col] = 0
        for row in range(1, rows):
            for col in range(1, cols):
                if matrix[row][0] == 0 or matrix[0][col] == 0:
                    matrix[row][col] = 0
        if matrix[0][0] == 0:
            for col in range(cols):
                matrix[0][col] = 0
        if first_col_zero:
            for row in range(rows):
                matrix[row][0] = 0
```

#### 54\. 螺旋矩阵｜思路详解与标准代码

* [ ] ⭐⭐ **54\. 螺旋矩阵**：维护 `top/bottom/left/right`；走完一边就收缩，后两边前检查边界。

**思路：**看到按圈遍历矩阵，维护 top、bottom、left、right 四条边界。每走完一条边就收缩，遍历下边和左边前再次确认边界仍有效。

**易错点：**

- 单行或单列不能重复访问

- 后两条边必须检查边界

**复杂度：**时间 O\(mn\)，额外空间 O\(1\)（不计输出）

```python
from typing import List

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        answer = []
        while top <= bottom and left <= right:
            for col in range(left, right + 1):
                answer.append(matrix[top][col])
            top += 1
            for row in range(top, bottom + 1):
                answer.append(matrix[row][right])
            right -= 1
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    answer.append(matrix[bottom][col])
                bottom -= 1
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    answer.append(matrix[row][left])
                left += 1
        return answer
```

#### 48\. 旋转图像｜思路详解与标准代码

* [ ] ⭐⭐ **48\. 旋转图像**：先沿主对角线转置，再逐行反转（顺时针 90°）。

**思路：**看到方阵原地顺时针旋转 90°，可分解为主对角线转置后逐行反转。两个操作都在原矩阵上执行。

**易错点：**

- 仅适用于方阵

- 方法原地修改且不返回矩阵

**复杂度：**时间 O\(n²\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()
```

#### 240\. 搜索二维矩阵 II｜思路详解与标准代码

* [ ] ⭐⭐ **240\. 搜索二维矩阵 II**：从右上角走；目标小则左，目标大则下，每步排除一行/列。

**思路：**看到每行每列递增，从右上角开始阶梯搜索。当前值过大就左移、过小就下移，每步排除一整行或列。

**易错点：**

- 空矩阵返回 False

- 不要误用第 74 题的一维整体有序条件

**复杂度：**时间 O\(m\+n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        row, col = 0, len(matrix[0]) - 1
        while row < len(matrix) and col >= 0:
            value = matrix[row][col]
            if value == target:
                return True
            if value > target:
                col -= 1
            else:
                row += 1
        return False
```

### 七、链表（14）

#### 160\. 相交链表｜思路详解与标准代码

* [ ] ⭐⭐ **160\. 相交链表**：`a` 到尾转 `headB`，`b` 到尾转 `headA`；走过相同总路程后相遇。

**思路：**看到两个单链表找引用相同的交点，用双指针消除长度差。a 走完 A 后转 B，b 走完 B 后转 A，最终在交点或 None 相遇。

**易错点：**

- 比较节点身份而非节点值

- 无交点时二者会同时为 None

**复杂度：**时间 O\(m\+n\)，额外空间 O\(1\)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a, b = headA, headB
        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a
```

#### 206\. 反转链表｜思路详解与标准代码

* [ ] ⭐⭐⭐ **206\. 反转链表**：保存 `next`，改 `cur.next=prev`，整体前进；三个指针必须能默写。

**思路：**看到单链表原地反转，维护 prev、cur、next 三个指针。先保存后继，再反向当前指针，最后整体前进。

**易错点：**

- 必须先保存 next

- 返回新的头 prev

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        dummy = ListNode()
        curr = head
        while curr:
            next = curr.next
            curr.next = dummy.next
            dummy.next = curr
            curr = next
        return dummy.next
```

#### 234\. 回文链表｜思路详解与标准代码

* [ ] ⭐⭐ **234\. 回文链表**：快慢找中点，反转后半段，再逐个比较；若要求无副作用则恢复。

**思路：**看到链表回文且要求常数空间，快慢指针找中点并反转后半段。逐节点比较前半段与反转后的后半段；实现可在结束后恢复链表。

**易错点：**

- 奇数长度跳过中点也可直接反转 slow 后比较

- 比较节点值

- 注意空链表

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        pre, cur = None, slow.next
        while cur:
            t = cur.next
            cur.next = pre
            pre, cur = cur, t
        while pre:
            if pre.val != head.val:
                return False
            pre, head = pre.next, head.next
        return True
```

#### 141\. 环形链表｜思路详解与标准代码

* [ ] ⭐⭐ **141\. 环形链表**：快慢指针；快指针追上慢指针即有环。

**思路：**看到判断链表是否有环，使用 Floyd 快慢指针。slow 每次一步、fast 每次两步，相遇则有环，fast 到尾则无环。

**易错点：**

- 循环条件要检查 fast 和 fast\.next

- 比较节点身份

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        s = set()
        while head:
            if head in s:
                return True
            s.add(head)
            head = head.next
        return False
```

#### 142\. 环形链表 II｜思路详解与标准代码

* [ ] ⭐⭐⭐ **142\. 环形链表 II**：快慢相遇后，头指针与相遇指针同速走，相遇点即入口。

**思路：**看到要返回环入口，先用快慢指针确认相遇。相遇后一个指针回到头部，两者同速前进，再次相遇处即入口。

**易错点：**

- 无环返回 None

- 第二阶段两个指针都每次走一步

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                ans = head
                while ans != slow:
                    ans = ans.next
                    slow = slow.next
                return ans
```

#### 21\. 合并两个有序链表｜思路详解与标准代码

* [ ] ⭐⭐ **21\. 合并两个有序链表**：虚拟头 \+ 尾指针，每次接较小节点，最后接剩余部分。

**思路：**看到两个有序链表合并，用虚拟头和尾指针逐个接入较小节点。任一链表耗尽后直接接上另一条剩余链。

**易错点：**

- 最后要接剩余部分

- 返回 dummy\.next

**复杂度：**时间 O\(m\+n\)，额外空间 O\(1\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        if list1 is None or list2 is None:
            return list1 or list2
        if list1.val <= list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
```

#### 2\. 两数相加｜思路详解与标准代码

* [ ] ⭐⭐ **2\. 两数相加**：同步遍历两链表和进位；循环条件包含 `carry`。

**思路：**看到逆序数字链表相加，按位同步遍历并维护进位。每轮把两个当前值和 carry 相加，生成个位节点并更新进位，直到链表和进位都为空。

**易错点：**

- 循环条件必须包含 carry

- 长短链表缺位按 0

- 返回虚拟头之后的链表

**复杂度：**时间 O\(max\(m,n\)\)，额外空间 O\(max\(m,n\)\)（结果链表）

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode()
        carry, curr = 0, dummy
        while l1 or l2 or carry:
            s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            carry, val = divmod(s, 10)
            curr.next = ListNode(val)
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
```

#### 19\. 删除倒数第 N 个结点｜思路详解与标准代码

* [ ] ⭐⭐⭐ **19\. 删除倒数第 N 个结点**：虚拟头；`fast` 先走 `n`，再同步，`slow` 停在待删节点前。

**思路：**看到删除倒数第 n 个节点，用虚拟头规避删除头节点特判。fast 先从虚拟头走 n\+1 步，再与 slow 同步，slow 最终停在待删节点前。

**易错点：**

- 虚拟头统一处理删除头节点

- 快慢间距需保证 slow 停在前驱

**复杂度：**时间 O\(L\)，额外空间 O\(1\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        fast = slow = dummy
        for _ in range(n):
            fast = fast.next
        while fast.next:
            slow, fast = slow.next, fast.next
        slow.next = slow.next.next
        return dummy.next
```

#### 24\. 两两交换链表节点｜思路详解与标准代码

* [ ] ⭐⭐ **24\. 两两交换链表节点**：虚拟头；每轮保存 `a,b,next`，重连 `prev→b→a→next`。

**思路：**看到按相邻两节点原地交换，用虚拟头保存每组前驱。每轮取 a、b 与下一组头，重连 prev→b→a→next，再把 prev 移到 a。

**易错点：**

- 交换的是节点而非值

- 奇数个节点末尾保持不变

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        t = self.swapPairs(head.next.next)
        p = head.next
        p.next = head
        head.next = t
        return p
```

#### 25\. K 个一组翻转链表｜思路详解与标准代码

* [ ] ⭐⭐⭐ **25\. K 个一组翻转链表**：先找第 `k` 个节点，不足则结束；保存组后节点，反转 `[groupPrev.next, groupNext)`。

**思路：**看到固定 k 个节点分组反转，先从 group\_prev 探测第 k 个节点，避免不足一组时误改。保存 group\_next 后反转半开区间 \[group\_prev\.next, group\_next\)，再把旧组头作为下一组前驱。

**易错点：**

- 不足 k 个必须原样保留

- 反转时 prev 初值应为 group\_next

- 每组结束后正确更新 group\_prev

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import Optional

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        group_prev = dummy
        while True:
            kth = group_prev
            for _ in range(k):
                kth = kth.next
                if kth is None:
                    return dummy.next
            group_next = kth.next
            prev, cur = group_next, group_prev.next
            while cur is not group_next:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
            old_group_head = group_prev.next
            group_prev.next = kth
            group_prev = old_group_head
```

#### 138\. 随机链表的复制｜思路详解与标准代码

* [ ] ⭐⭐ **138\. 随机链表的复制**：哈希 `old→new` 两遍连边；进阶是复制节点穿插原链表再拆分。

**思路：**看到节点含 next 和 random 两种引用，需要保持图结构映射。第一遍为每个旧节点创建新节点，第二遍通过 old→new 哈希连接 next 与 random。

**易错点：**

- None 映射要安全处理

- random 可能指向自身或任意节点

- 不能复用原节点

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import Optional

class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if head is None:
            return None
        copies = {None: None}
        cur = head
        while cur is not None:
            copies[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur is not None:
            copies[cur].next = copies[cur.next]
            copies[cur].random = copies[cur.random]
            cur = cur.next
        return copies[head]
```

#### 148\. 排序链表｜思路详解与标准代码

* [ ] ⭐⭐⭐ **148\. 排序链表**：快慢切半 \+ 归并排序；注意切断 `slow.next=None`。$O(n\log n)$。

**思路：**看到链表排序且要求 O\(n log n\)，使用归并排序。快慢指针切成两半并断链，递归排序后用虚拟头合并两条有序链表。

**易错点：**

- 切分后必须断开 slow\.next

- 递归基线是空或单节点

**复杂度：**时间 O\(n log n\)，递归栈额外空间 O\(log n\)

```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        l1, l2 = head, slow.next
        slow.next = None
        l1, l2 = self.sortList(l1), self.sortList(l2)
        dummy = ListNode()
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 or l2
        return dummy.next
```

#### 23\. 合并 K 个升序链表｜思路详解与标准代码

* [ ] ⭐⭐⭐ **23\. 合并 K 个升序链表**：最小堆存 `(值,序号,节点)` 或两两归并；每弹一个就推其后继。

**思路：**看到 k 条有序链表合并，用最小堆维护每条链当前头。堆元素加入唯一序号避免节点值相等时比较 ListNode；每弹出一个节点就推入它的后继。

**易错点：**

- 堆元组必须有可比较的唯一序号

- 空链表不要入堆

- 弹出节点后推进其后继

**复杂度：**时间 O\(N log k\)，额外空间 O\(k\)

```python
from typing import List, Optional
import heapq

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        sequence = 0
        for node in lists:
            if node is not None:
                heapq.heappush(heap, (node.val, sequence, node))
                sequence += 1
        dummy = ListNode(0)
        tail = dummy
        while heap:
            _, _, node = heapq.heappop(heap)
            tail.next = node
            tail = node
            if node.next is not None:
                heapq.heappush(heap, (node.next.val, sequence, node.next))
                sequence += 1
        return dummy.next
```

#### 146\. LRU 缓存｜思路详解与标准代码

* [ ] ⭐⭐⭐ **146\. LRU 缓存**：哈希定位 \+ 双向链表维护新旧；访问移到头，超容量删尾；所有操作 $O(1)$。

**思路：**看到 get/put 都要求 O\(1\) 且要淘汰最久未使用项，组合哈希表与双向链表。哈希定位节点，链表头侧表示最新、尾侧表示最旧；访问或更新移到头，超容量删除尾节点。

**易错点：**

- get 命中也必须更新新旧顺序

- put 已存在键不能增加 size

- 淘汰时同时删除哈希映射

**复杂度：**每次 get/put 时间 O\(1\)，额外空间 O\(capacity\)

```python
class Node:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_first(self, node: Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_first(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_first(node)
            return
        node = Node(key, value)
        self.cache[key] = node
        self._add_first(node)
        if len(self.cache) > self.capacity:
            oldest = self.tail.prev
            self._remove(oldest)
            del self.cache[oldest.key]
```

### 八、二叉树（15）

#### 94\. 二叉树的中序遍历｜思路详解与标准代码

* [ ] ⭐⭐⭐ **94\. 二叉树的中序遍历**：迭代时一路压左；弹出访问，再转右。BST 中序有序。

**思路：**看到中序遍历，迭代栈模拟“左—根—右”。当前节点一路压左，弹栈访问后转向右子树。

**易错点：**

- 外层条件是 root 或 stack

- BST 中序严格有序仅在键不重复时成立

**复杂度：**时间 O\(n\)，额外空间 O\(h\)

```python
from typing import List, Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root):
            if root is None:
                return
            dfs(root.left)
            ans.append(root.val)
            dfs(root.right)

        ans = []
        dfs(root)
        return ans
```

#### 104\. 二叉树的最大深度｜思路详解与标准代码

* [ ] ⭐ **104\. 二叉树的最大深度**：`1+max(left,right)`，空树 0。

**思路：**看到树高，定义 DFS 返回当前子树最大深度。空节点返回 0，非空节点返回 1 加左右子树深度较大值。

**易错点：**

- 空树返回 0

- 深度包含当前节点

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        l, r = self.maxDepth(root.left), self.maxDepth(root.right)
        return 1 + max(l, r)
```

#### 226\. 翻转二叉树｜思路详解与标准代码

* [ ] ⭐ **226\. 翻转二叉树**：递归交换左右子树；返回根。

**思路：**看到镜像翻转，递归处理左右子树并交换。每个节点只交换一次，最终返回原根节点。

**易错点：**

- 空节点直接返回 None

- 交换的是子树引用

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        l, r = self.invertTree(root.left), self.invertTree(root.right)
        root.left, root.right = r, l
        return root
```

#### 101\. 对称二叉树｜思路详解与标准代码

* [ ] ⭐⭐ **101\. 对称二叉树**：递归比较 `(left.left,right.right)` 与 `(left.right,right.left)`。

**思路：**看到判断镜像对称，递归比较一对镜像位置节点。两者都空为真、仅一方空或值不同为假，随后交叉比较外侧与内侧子树。

**易错点：**

- 必须交叉比较

- 空树视为对称

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def dfs(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
            if root1 == root2:
                return True
            if root1 is None or root2 is None or root1.val != root2.val:
                return False
            return dfs(root1.left, root2.right) and dfs(root1.right, root2.left)

        return dfs(root.left, root.right)
```

#### 543\. 二叉树的直径｜思路详解与标准代码

* [ ] ⭐⭐⭐ **543\. 二叉树的直径**：DFS 返回向下最大深度；全局更新 `leftDepth+rightDepth`。

**思路：**看到任意两节点最长路径，后序 DFS 返回向下最大深度。每个节点以 left\+right 更新经过它的最长边数，再向父节点返回 1\+max\(left,right\)。

**易错点：**

- 题目答案按边数而非节点数

- 全局答案初值为 0

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def dfs(root: Optional[TreeNode]) -> int:
            if root is None:
                return 0
            l, r = dfs(root.left), dfs(root.right)
            nonlocal ans
            ans = max(ans, l + r)
            return 1 + max(l, r)

        ans = 0
        dfs(root)
        return ans
```

#### 102\. 二叉树的层序遍历｜思路详解与标准代码

* [ ] ⭐⭐⭐ **102\. 二叉树的层序遍历**：队列；每轮固定读取当前 `len(q)` 个节点形成一层。

**思路：**看到按层输出，使用队列 BFS。每轮固定读取进入该轮时的队列长度，将这些节点组成一层并把孩子入队。

**易错点：**

- 空树返回空列表

- 每轮必须固定层大小

**复杂度：**时间 O\(n\)，额外空间 O\(w\)，w 为最大层宽

```python
from typing import List, Optional
from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        if root is None:
            return ans
        q = deque([root])
        while q:
            t = []
            for _ in range(len(q)):
                node = q.popleft()
                t.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(t)
        return ans
```

#### 108\. 将有序数组转换为二叉搜索树｜思路详解与标准代码

* [ ] ⭐ **108\. 将有序数组转换为 BST**：取中点为根，递归左右半区；天然高度平衡。

**思路：**看到有序数组构造高度平衡 BST，递归取区间中点为根。中点左右区间分别构造左右子树，区间为空返回 None。

**易错点：**

- 边界应统一为闭区间或半开区间

- 偶数长度任选中间节点都可

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(log n\)

```python
from typing import List, Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(l: int, r: int) -> Optional[TreeNode]:
            if l > r:
                return None
            mid = (l + r) >> 1
            return TreeNode(nums[mid], dfs(l, mid - 1), dfs(mid + 1, r))

        return dfs(0, len(nums) - 1)
```

#### 98\. 验证二叉搜索树｜思路详解与标准代码

* [ ] ⭐⭐⭐ **98\. 验证 BST**：递归传严格上下界，或中序必须严格递增；不能只比较父子。

**思路：**看到验证整棵 BST，递归传递每个节点允许的严格上下界。节点值必须落在 \(low, high\) 内，左子树收紧上界，右子树收紧下界。

**易错点：**

- 必须使用严格不等号

- 不能只比较父子节点

- 初始边界可用正负无穷

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
from typing import Optional
from math import inf

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(root: Optional[TreeNode]) -> bool:
            if root is None:
                return True
            if not dfs(root.left):
                return False
            nonlocal prev
            if prev >= root.val:
                return False
            prev = root.val
            return dfs(root.right)

        prev = -inf
        return dfs(root)
```

#### 230\. 二叉搜索树中第 K 小的元素｜思路详解与标准代码

* [ ] ⭐⭐ **230\. BST 第 K 小**：中序遍历计数，第 `k` 个即答案；可提前退出。

**思路：**看到 BST 第 k 小，利用中序遍历按升序访问。迭代栈每弹出一个节点就递减 k，k 为 0 时立即返回该值。

**易错点：**

- k 从 1 开始

- 利用提前返回避免遍历全树

**复杂度：**时间 O\(h\+k\)，额外空间 O\(h\)

```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stk = []
        while root or stk:
            if root:
                stk.append(root)
                root = root.left
            else:
                root = stk.pop()
                k -= 1
                if k == 0:
                    return root.val
                root = root.right
```

#### 199\. 二叉树的右视图｜思路详解与标准代码

* [ ] ⭐⭐ **199\. 二叉树右视图**：层序每层取最后一个；或 DFS 先右后左、首次到达该深度记录。

**思路：**看到每层最右节点，使用层序 BFS。每层固定处理当前节点数，并把该层最后一个出队节点的值加入答案。

**易错点：**

- 空树返回空列表

- 取的是每层最后一个而非全局最右坐标

**复杂度：**时间 O\(n\)，额外空间 O\(w\)

```python
from typing import List, Optional
from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        if root is None:
            return ans
        q = deque([root])
        while q:
            ans.append(q[0].val)
            for _ in range(len(q)):
                node = q.popleft()
                if node.right:
                    q.append(node.right)
                if node.left:
                    q.append(node.left)
        return ans
```

#### 114\. 二叉树展开为链表｜思路详解与标准代码

* [ ] ⭐⭐ **114\. 二叉树展开为链表**：逆前序（右→左→根），令 `root.right=prev, root.left=None`。

**思路：**看到按前序原地展开，使用逆前序“右—左—根”递归。维护已展开后缀 prev，处理当前节点时令 right 指向 prev、left 置空，再更新 prev。

**易错点：**

- 必须先递归右再递归左

- 每个 left 最终都为 None

- 方法原地修改

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while root:
            if root.left:
                pre = root.left
                while pre.right:
                    pre = pre.right
                pre.right = root.right
                root.right = root.left
                root.left = None
            root = root.right
```

#### 105\. 从前序与中序遍历序列构造二叉树｜思路详解与标准代码

* [ ] ⭐⭐⭐ **105\. 前序与中序构造二叉树**：前序首个为根；哈希定位中序根，按左子树长度切分。

**思路：**看到前序和中序且节点值互异，用哈希记录中序位置。递归状态用前序根下标和中序闭区间；由根在中序中的位置计算左子树长度，再定位两个子树。

**易错点：**

- 题目保证节点值唯一

- 切片会引入 O\(n²\) 开销，应传下标

- 空区间返回 None

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List, Optional

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        index = {value: i for i, value in enumerate(inorder)}

        def build(pre_root: int, in_left: int, in_right: int) -> Optional[TreeNode]:
            if in_left > in_right:
                return None
            root_value = preorder[pre_root]
            in_root = index[root_value]
            left_size = in_root - in_left
            root = TreeNode(root_value)
            root.left = build(pre_root + 1, in_left, in_root - 1)
            root.right = build(pre_root + left_size + 1, in_root + 1, in_right)
            return root

        return build(0, 0, len(inorder) - 1)
```

#### 437\. 路径总和 III｜思路详解与标准代码

* [ ] ⭐⭐⭐ **437\. 路径总和 III**：树上前缀和；进入节点先查 `pre-target`，计数 \+1，离开时必须 \-1。

**思路：**看到路径必须向下但可从任意节点开始，使用树上前缀和频次。进入节点后累加前缀和，用 count\[pre\-target\] 统计以当前节点结尾的路径；递归返回前必须撤销当前前缀和。

**易错点：**

- 初始化 count\[0\]=1

- 先查询再增加当前前缀和

- 回溯时必须减一避免串到兄弟子树

**复杂度：**时间 O\(n\)，额外空间 O\(h\)

```python
from typing import Optional
from collections import defaultdict

class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        count = defaultdict(int)
        count[0] = 1

        def dfs(node: Optional[TreeNode], prefix: int) -> int:
            if node is None:
                return 0
            prefix += node.val
            answer = count[prefix - targetSum]
            count[prefix] += 1
            answer += dfs(node.left, prefix)
            answer += dfs(node.right, prefix)
            count[prefix] -= 1
            return answer

        return dfs(root, 0)
```

#### 236\. 二叉树的最近公共祖先｜思路详解与标准代码

* [ ] ⭐⭐⭐ **236\. 二叉树最近公共祖先**：左右递归返回目标；若左右都非空当前即 LCA，否则返回非空一侧。

**思路：**看到普通二叉树两节点 LCA，用后序递归让函数返回当前子树找到的目标或祖先。当前为 p/q 就返回自身；左右都非空则当前为 LCA，否则向上传递非空侧。

**易错点：**

- 比较节点身份

- 题目保证 p、q 存在

- 不能套用 BST 大小关系

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        return root if left and right else (left or right)
```

#### 124\. 二叉树中的最大路径和｜思路详解与标准代码

* [ ] ⭐⭐⭐ **124\. 二叉树最大路径和**：DFS 返回“从当前向下单边最大贡献”；负贡献取 0，全局更新 `node+l+r`。

**思路：**看到路径可从任意节点起止但不能分叉后再向父延伸，后序 DFS 返回单边最大贡献。左右负贡献截为 0，以 node\+left\+right 更新全局答案，再返回 node\+max\(left,right\)。

**易错点：**

- 全局答案应初始化为负无穷以处理全负树

- 返回父节点时只能选一侧

- 节点值必须计入路径

**复杂度：**时间 O\(n\)，递归栈额外空间 O\(h\)

```python
from typing import Optional

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        answer = float('-inf')

        def gain(node: Optional[TreeNode]) -> int:
            nonlocal answer
            if node is None:
                return 0
            left = max(gain(node.left), 0)
            right = max(gain(node.right), 0)
            answer = max(answer, node.val + left + right)
            return node.val + max(left, right)

        gain(root)
        return int(answer)
```

### 九、图论（4）

#### 200\. 岛屿数量｜思路详解与标准代码

* [ ] ⭐⭐⭐ **200\. 岛屿数量**：遇陆地答案 \+1，从它 DFS/BFS 淹没整个连通块；注意边界和访问标记。

**思路：**看到网格连通块计数，遍历每个格子，遇到未访问陆地就增加答案并 DFS 淹没整个岛。访问时直接把陆地改成水，避免额外 visited。

**易错点：**

- 先做边界检查

- 每块陆地只访问一次

- 方法会修改输入网格

**复杂度：**时间 O\(mn\)，递归栈最坏额外空间 O\(mn\)

```python
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        answer = 0

        def dfs(row: int, col: int) -> None:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return
            if grid[row][col] != '1':
                return
            grid[row][col] = '0'
            dfs(row + 1, col)
            dfs(row - 1, col)
            dfs(row, col + 1)
            dfs(row, col - 1)

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == '1':
                    answer += 1
                    dfs(row, col)
        return answer
```

#### 994\. 腐烂的橘子｜思路详解与标准代码

* [ ] ⭐⭐ **994\. 腐烂的橘子**：多源 BFS；所有腐橘同时入队，每层一分钟，最后检查是否仍有新鲜橘。

**思路：**看到多个源点同步扩散，用多源 BFS。所有腐橘先入队并统计新鲜橘，每轮传播一层代表一分钟；新鲜数归零时返回分钟数，否则无解。

**易错点：**

- 无新鲜橘应返回 0

- 分钟只在本层实际腐烂新橘时增加

- 最终仍有新鲜橘返回 \-1

**复杂度：**时间 O\(mn\)，额外空间 O\(mn\)

```python
from typing import List
from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 2:
                    queue.append((row, col))
                elif grid[row][col] == 1:
                    fresh += 1
        minutes = 0
        while queue and fresh > 0:
            for _ in range(len(queue)):
                row, col = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh -= 1
                        queue.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```

#### 207\. 课程表｜思路详解与标准代码

* [ ] ⭐⭐⭐ **207\. 课程表**：建图与入度；Kahn 拓扑 BFS，最终处理节点数等于课程数则无环。

**思路：**看到先修关系能否完成，本质是检测有向图是否有环。建立从先修课到后续课的邻接表与入度，Kahn BFS 不断移除入度为 0 的节点，处理数等于课程数则可行。

**易错点：**

- 边方向应为 prerequisite→course

- 孤立课程也计入总数

- 最终比较处理节点数

**复杂度：**时间 O\(V\+E\)，额外空间 O\(V\+E\)

```python
from typing import List

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indeg[a] += 1
        q = [i for i, x in enumerate(indeg) if x == 0]
        for i in q:
            numCourses -= 1
            for j in g[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)
        return numCourses == 0
```

#### 208\. 实现 Trie（前缀树）｜思路详解与标准代码

* [ ] ⭐⭐⭐ **208\. 实现 Trie**：节点含 `children` 与 `is_end`；插入逐字符建边，查前缀不要求 `is_end`。

**思路：**看到大量单词插入、整词查询和前缀查询，使用 Trie 逐字符共享路径。每个节点保存 children 映射和结束标记；search 需检查结束标记，startsWith 只需路径存在。

**易错点：**

- 整词与前缀语义不同

- 插入结束必须标记 is\_end

- 空前缀应能匹配

**复杂度：**每次操作时间 O\(L\)，额外空间 O\(所有插入字符总数\)

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def _find(self, text: str):
        node = self
        for ch in text:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node
```

### 十、回溯（8）

#### 46\. 全排列｜思路详解与标准代码

* [ ] ⭐⭐⭐ **46\. 全排列**：路径长度为 `n` 收集；每层枚举未使用元素，`used` 标记后恢复。

**思路：**看到使用全部互异元素的所有排列，回溯维护路径与 used 标记。每层选择一个未使用元素，路径长度达到 n 时复制收集，返回后恢复状态。

**易错点：**

- 收集时复制 path

- 回溯必须恢复 used 和路径

- 题目元素互异

**复杂度：**时间 O\(n·n\!\)，额外空间 O\(n\)（不计输出）

```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(i: int):
            if i >= n:
                ans.append(t[:])
                return
            for j, x in enumerate(nums):
                if not vis[j]:
                    vis[j] = True
                    t[i] = x
                    dfs(i + 1)
                    vis[j] = False

        n = len(nums)
        vis = [False] * n
        t = [0] * n
        ans = []
        dfs(0)
        return ans
```

#### 78\. 子集｜思路详解与标准代码

* [ ] ⭐⭐⭐ **78\. 子集**：每个递归节点都收集；从 `start` 往后选，天然避免重复位置。

**思路：**看到互异数组的所有子集，回溯用 start 限制后续选择。每个递归节点都代表一个合法子集并立即复制收集，再枚举后续元素。

**易错点：**

- 空集必须包含

- 使用 start 避免同一组合不同顺序

- 收集时复制路径

**复杂度：**时间 O\(n·2ⁿ\)，额外空间 O\(n\)（不计输出）

```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(i: int):
            if i == len(nums):
                ans.append(t[:])
                return
            dfs(i + 1)
            t.append(nums[i])
            dfs(i + 1)
            t.pop()

        ans = []
        t = []
        dfs(0)
        return ans
```

#### 17\. 电话号码的字母组合｜思路详解与标准代码

* [ ] ⭐⭐ **17\. 电话号码的字母组合**：第 `i` 位枚举映射中的字母；到 `len(digits)` 收集。

**思路：**看到每位数字对应若干字母，按数字位置回溯。第 i 层枚举该数字映射的每个字母，走到末尾时拼接路径收集。

**易错点：**

- 空 digits 返回空列表而非含空串

- 映射仅覆盖 2 到 9

**复杂度：**时间 O\(4ⁿ·n\)，额外空间 O\(n\)（不计输出）

```python
from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        d = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        ans = [""]
        for i in digits:
            s = d[int(i) - 2]
            ans = [a + b for a in ans for b in s]
        return ans
```

#### 39\. 组合总和｜思路详解与标准代码

* [ ] ⭐⭐ **39\. 组合总和**：排序可剪枝；递归仍传 `i` 表示同一数字可重复选。

**思路：**看到候选数可重复使用且组合不计顺序，排序后以 start 回溯。每层从 start 向后选数，选中后仍传当前下标；当前数超过剩余目标时剪枝。

**易错点：**

- 重复使用时递归传 i 而非 i\+1

- 用 start 避免排列重复

- 题目候选数互异

**复杂度：**最坏时间 O\(n^\(target/min\)\)，递归栈额外空间 O\(target/min\)

```python
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(i: int, s: int):
            if s == 0:
                ans.append(t[:])
                return
            if s < candidates[i]:
                return
            for j in range(i, len(candidates)):
                t.append(candidates[j])
                dfs(j, s - candidates[j])
                t.pop()

        candidates.sort()
        t = []
        ans = []
        dfs(0, target)
        return ans
```

#### 22\. 括号生成｜思路详解与标准代码

* [ ] ⭐⭐⭐ **22\. 括号生成**：左括号数 `<n` 可加左；右括号数 `<左括号数` 才可加右。

**思路：**看到生成所有合法括号串，回溯维护已用左右括号数。left\<n 时可加左括号，right\<left 时才可加右括号，长度达到 2n 时收集。

**易错点：**

- 任何前缀都必须满足 right\<=left

- 收集条件为长度 2n

**复杂度：**时间 O\(Cn·n\)，额外空间 O\(n\)（不计输出），Cn 为第 n 个卡特兰数

```python
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(l: int, r: int, t: str):
            if l > n or r > n or l < r:
                return
            if l == n and r == n:
                ans.append(t)
                return
            dfs(l + 1, r, t + "(")
            dfs(l, r + 1, t + ")")

        ans = []
        dfs(0, 0, "")
        return ans
```

#### 79\. 单词搜索｜思路详解与标准代码

* [ ] ⭐⭐⭐ **79\. 单词搜索**：从每格启动 DFS；字符匹配后临时标记访问，四向搜索，返回前恢复。

**思路：**看到网格中按四方向且单格不可重复组成单词，从每个可能起点 DFS。匹配当前字符后临时标记该格，递归四邻，失败返回前恢复。

**易错点：**

- 同一路径不能重复用格子

- 成功提前返回时也应保持实现语义安全

- 空网格需防越界

**复杂度：**时间 O\(mn·3^L\)，递归栈额外空间 O\(L\)

```python
from typing import List

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word:
            return True
        if not board or not board[0]:
            return False
        rows, cols = len(board), len(board[0])

        def dfs(row: int, col: int, index: int) -> bool:
            if board[row][col] != word[index]:
                return False
            if index == len(word) - 1:
                return True
            saved = board[row][col]
            board[row][col] = '#'
            found = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                    if dfs(nr, nc, index + 1):
                        found = True
                        break
            board[row][col] = saved
            return found

        for row in range(rows):
            for col in range(cols):
                if dfs(row, col, 0):
                    return True
        return False
```

#### 131\. 分割回文串｜思路详解与标准代码

* [ ] ⭐⭐ **131\. 分割回文串**：枚举下一段终点，只有子串为回文才递归；到字符串末尾收集。

**思路：**看到把字符串分成所有回文片段，回溯枚举下一段终点。只有 s\[start:end\] 为回文时才加入路径并继续，start 到末尾时复制答案。

**易错点：**

- 终点边界要包含到 len\(s\)

- 收集时复制路径

- 空串有一个空分割

**复杂度：**最坏时间 O\(n·2ⁿ\)，递归栈额外空间 O\(n\)

```python
from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def dfs(i: int):
            if i == n:
                ans.append(t[:])
                return
            for j in range(i, n):
                if f[i][j]:
                    t.append(s[i : j + 1])
                    dfs(j + 1)
                    t.pop()

        n = len(s)
        f = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                f[i][j] = s[i] == s[j] and f[i + 1][j - 1]
        ans = []
        t = []
        dfs(0)
        return ans
```

#### 51\. N 皇后｜思路详解与标准代码

* [ ] ⭐⭐⭐ **51\. N 皇后**：逐行放置；集合维护列、`row-col`、`row+col`，放置后回溯恢复。

**思路：**看到每行每列及两条对角线不能冲突，逐行回溯放皇后。用列、row\-col、row\+col 三个集合维护占用，放置后递归下一行并恢复。

**易错点：**

- 两类对角线标识不能混淆

- 每行恰放一个

- 回溯要同步恢复棋盘和集合

**复杂度：**时间 O\(n\!\)，额外空间 O\(n²\)（棋盘与递归状态）

```python
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def dfs(i: int):
            if i == n:
                ans.append(["".join(row) for row in g])
                return
            for j in range(n):
                if col[j] + dg[i + j] + udg[n - i + j] == 0:
                    g[i][j] = "Q"
                    col[j] = dg[i + j] = udg[n - i + j] = 1
                    dfs(i + 1)
                    col[j] = dg[i + j] = udg[n - i + j] = 0
                    g[i][j] = "."

        ans = []
        g = [["."] * n for _ in range(n)]
        col = [0] * n
        dg = [0] * (n << 1)
        udg = [0] * (n << 1)
        dfs(0)
        return ans
```

### 十一、二分查找（6）

#### 35\. 搜索插入位置｜思路详解与标准代码

* [ ] ⭐⭐⭐ **35\. 搜索插入位置**：找第一个 `>= target` 的下标，即标准左边界。

**思路：**看到有序数组中查找或插入位置，求第一个大于等于 target 的下标。维护半开区间 \[left,right\)，根据 nums\[mid\] 是否小于 target 丢弃一半，最终 left 即答案。

**易错点：**

- 右边界可取 len\(nums\)

- 空数组返回 0

- 循环与返回边界模板要一致

**复杂度：**时间 O\(log n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)
        while l < r:
            mid = (l + r) >> 1
            if nums[mid] >= target:
                r = mid
            else:
                l = mid + 1
        return l
```

#### 74\. 搜索二维矩阵｜思路详解与标准代码

* [ ] ⭐⭐ **74\. 搜索二维矩阵**：把矩阵视为长度 `m*n` 的一维有序数组，`mid→[mid//n][mid%n]`。

**思路：**看到矩阵每行有序且下一行首元素大于上一行末元素，可视为整体一维有序数组。对 \[0,mn\) 二分，通过 mid//n 和 mid%n 映射回行列。

**易错点：**

- 空矩阵返回 False

- 列数 n 用于下标映射

- 不要与第 240 题条件混淆

**复杂度：**时间 O\(log\(mn\)\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        rows, cols = len(matrix), len(matrix[0])
        left, right = 0, rows * cols
        while left < right:
            mid = (left + right) // 2
            value = matrix[mid // cols][mid % cols]
            if value < target:
                left = mid + 1
            else:
                right = mid
        return left < rows * cols and matrix[left // cols][left % cols] == target
```

#### 34\. 在排序数组中查找元素的第一个和最后一个位置｜思路详解与标准代码

* [ ] ⭐⭐⭐ **34\. 在排序数组中查找首尾位置**：左端=`lower_bound(target)`；右端=`lower_bound(target+1)-1`。

**思路：**看到有序数组目标区间，分别寻找 target 的左边界与第一个大于 target 的位置。若左边界越界或不等于 target 返回 \[\-1,\-1\]，否则右端为 upper\_bound\-1。

**易错点：**

- 不能用 target\+1 规避整型边界，直接写 upper\_bound 更稳

- 先验证目标是否存在

**复杂度：**时间 O\(log n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def lower_bound(value: int) -> int:
            left, right = 0, len(nums)
            while left < right:
                mid = (left + right) // 2
                if nums[mid] < value:
                    left = mid + 1
                else:
                    right = mid
            return left

        def upper_bound(value: int) -> int:
            left, right = 0, len(nums)
            while left < right:
                mid = (left + right) // 2
                if nums[mid] <= value:
                    left = mid + 1
                else:
                    right = mid
            return left

        first = lower_bound(target)
        if first == len(nums) or nums[first] != target:
            return [-1, -1]
        return [first, upper_bound(target) - 1]
```

#### 33\. 搜索旋转排序数组｜思路详解与标准代码

* [ ] ⭐⭐⭐ **33\. 搜索旋转排序数组**：每轮至少一半有序；判断目标是否落在有序半边，保留对应区间。

**思路：**看到无重复旋转有序数组，每轮至少有一半仍有序。判断 mid 落在哪个有序半边，再检查 target 是否位于该半边范围，保留可能区间。

**易错点：**

- 题目无重复元素

- 区间比较的等号必须统一

- 找不到返回 \-1

**复杂度：**时间 O\(log n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left < right:
            mid = (left + right) >> 1
            if nums[0] <= nums[mid]:
                if nums[0] <= target <= nums[mid]:
                    right = mid
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[n - 1]:
                    left = mid + 1
                else:
                    right = mid
        return left if nums[left] == target else -1
```

#### 153\. 寻找旋转排序数组中的最小值｜思路详解与标准代码

* [ ] ⭐⭐ **153\. 寻找旋转排序数组中的最小值**：比较 `nums[mid]` 与 `nums[right]`；大于则最小值在右，否则含 mid 的左侧。

**思路：**看到无重复旋转数组求最小值，用 mid 与 right 比较。若 nums\[mid\]\>nums\[right\]，最小值严格在 mid 右侧；否则最小值在含 mid 的左半边。

**易错点：**

- 右侧更新为 mid 而非 mid\-1

- 题目数组非空且无重复

**复杂度：**时间 O\(log n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) >> 1
            if nums[mid] > nums[-1]:
                l = mid + 1
            else:
                r = mid
        return nums[l]
```

#### 4\. 寻找两个正序数组的中位数｜思路详解与标准代码

* [ ] ⭐ **4\. 寻找两个正序数组的中位数**：在短数组二分切分点，使左半元素数固定且 `Aleft≤Bright, Bleft≤Aright`。

**思路：**看到两个有序数组且要求对数时间，应在较短数组上二分分割线。保持左半总元素数固定，寻找 Aleft\<=Bright 且 Bleft\<=Aright 的分割；根据总长度奇偶返回左侧最大值或两侧中间值均值。

**易错点：**

- 必须保证在较短数组二分

- 分割到边界时使用正负无穷

- 偶数长度返回浮点均值

**复杂度：**时间 O\(log\(min\(m,n\)\)\)，额外空间 O\(1\)

```python
from typing import List
from math import inf

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        left_size = (m + n + 1) // 2
        left, right = 0, m
        while left <= right:
            i = (left + right) // 2
            j = left_size - i
            a_left = nums1[i - 1] if i > 0 else -inf
            a_right = nums1[i] if i < m else inf
            b_left = nums2[j - 1] if j > 0 else -inf
            b_right = nums2[j] if j < n else inf
            if a_left <= b_right and b_left <= a_right:
                if (m + n) % 2:
                    return float(max(a_left, b_left))
                return (max(a_left, b_left) + min(a_right, b_right)) / 2.0
            if a_left > b_right:
                right = i - 1
            else:
                left = i + 1
        raise ValueError("Input arrays must be sorted")
```

### 十二、栈（5）

#### 20\. 有效的括号｜思路详解与标准代码

* [ ] ⭐ **20\. 有效的括号**：右括号到来时检查栈顶匹配；最后栈必须空。

**思路：**看到括号嵌套合法性，使用栈保存尚未匹配的左括号。遇右括号时栈不能为空且栈顶必须对应，扫描结束后栈也必须为空。

**易错点：**

- 右括号到来时先判空

- 最终残留左括号也不合法

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stk = []
        d = {'()', '[]', '{}'}
        for c in s:
            if c in '({[':
                stk.append(c)
            elif not stk or stk.pop() + c not in d:
                return False
        return not stk
```

#### 155\. 最小栈｜思路详解与标准代码

* [ ] ⭐⭐ **155\. 最小栈**：数据栈 \+ 最小栈；最小栈压入当前最小值，弹出同步。

**思路：**看到栈操作与取最小值都要求 O\(1\)，为每个入栈元素同步保存当时最小值。push 同时压入值和当前最小值，pop 同步弹出，getMin 读取辅助栈顶。

**易错点：**

- 重复最小值也必须同步压栈

- 题目保证查询时栈非空

**复杂度：**每次操作时间 O\(1\)，额外空间 O\(n\)

```python
from math import inf

class MinStack:
    def __init__(self):
        self.stk1 = []
        self.stk2 = [inf]

    def push(self, val: int) -> None:
        self.stk1.append(val)
        self.stk2.append(min(val, self.stk2[-1]))

    def pop(self) -> None:
        self.stk1.pop()
        self.stk2.pop()

    def top(self) -> int:
        return self.stk1[-1]

    def getMin(self) -> int:
        return self.stk2[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

#### 394\. 字符串解码｜思路详解与标准代码

* [ ] ⭐⭐ **394\. 字符串解码**：遇 `[` 保存当前次数和字符串并重置；遇 `]` 出栈并展开拼接。

**思路：**看到 k\[encoded\] 可嵌套，扫描字符串并用栈保存进入括号前的字符串和重复次数。数字连续累积，遇 \[ 入栈并重置，遇 \] 出栈组合，字母追加到当前串。

**易错点：**

- 重复次数可能多位

- 嵌套时保存次数和前缀

- 括号内字符串可含多段

**复杂度：**时间 O\(展开后字符串长度\)，额外空间 O\(嵌套深度与输出长度\)

```python
class Solution:
    def decodeString(self, s: str) -> str:
        s1, s2 = [], []
        num, res = 0, ''
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == '[':
                s1.append(num)
                s2.append(res)
                num, res = 0, ''
            elif c == ']':
                res = s2.pop() + res * s1.pop()
            else:
                res += c
        return res
```

#### 739\. 每日温度｜思路详解与标准代码

* [ ] ⭐⭐⭐ **739\. 每日温度**：递减栈存未找到更高温度的下标；新温度更高时弹出并填距离。

**思路：**看到每个位置右侧第一个更大值距离，使用存下标的单调递减栈。新温度高于栈顶对应温度时不断弹栈，并用当前下标差填写答案。

**易错点：**

- 严格更高才弹，相等不能结算

- 栈存下标而非温度

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stk = []
        n = len(temperatures)
        ans = [0] * n
        for i in range(n - 1, -1, -1):
            while stk and temperatures[stk[-1]] <= temperatures[i]:
                stk.pop()
            if stk:
                ans[i] = stk[-1] - i
            stk.append(i)
        return ans
```

#### 84\. 柱状图中最大的矩形｜思路详解与标准代码

* [ ] ⭐⭐⭐ **84\. 柱状图中最大矩形**：递增栈 \+ 两端哨兵 0；弹出高度时，宽度为 `i-st[-1]-1`。

**思路：**看到每根柱向左右延伸到首个更矮柱，使用单调递增下标栈。给两端加 0 哨兵；遇更矮高度时弹出柱高，其可用宽度为 i\-stack\[\-1\]\-1，并更新面积。

**易错点：**

- 栈必须存下标

- 弹出后的新栈顶才是左侧更矮边界

- 尾部哨兵负责清空剩余柱子

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        values = [0] + heights + [0]
        stack = [0]
        answer = 0
        for i in range(1, len(values)):
            while values[stack[-1]] > values[i]:
                height = values[stack.pop()]
                width = i - stack[-1] - 1
                answer = max(answer, height * width)
            stack.append(i)
        return answer
```

### 十三、堆（3）

#### 215\. 数组中的第 K 个最大元素｜思路详解与标准代码

* [ ] ⭐⭐⭐ **215\. 数组中的第 K 个最大元素**：大小为 `k` 的最小堆；或 Quickselect，只递归包含目标下标的一侧。

**思路：**看到第 k 大且无需完整排序，用大小为 k 的最小堆。依次加入元素，堆超过 k 就弹出最小值，最终堆顶是第 k 大。

**易错点：**

- k 合法且从 1 开始

- 最小堆维护最大的 k 个元素

**复杂度：**时间 O\(n log k\)，额外空间 O\(k\)

```python
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_sort(l: int, r: int) -> int:
            if l == r:
                return nums[l]
            i, j = l - 1, r + 1
            x = nums[(l + r) >> 1]
            while i < j:
                while 1:
                    i += 1
                    if nums[i] >= x:
                        break
                while 1:
                    j -= 1
                    if nums[j] <= x:
                        break
                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]
            if j < k:
                return quick_sort(j + 1, r)
            return quick_sort(l, j)

        n = len(nums)
        k = n - k
        return quick_sort(0, n - 1)
```

#### 347\. 前 K 个高频元素｜思路详解与标准代码

* [ ] ⭐⭐ **347\. 前 K 个高频元素**：计数后用大小为 `k` 的最小堆，或按频次桶排序。

**思路：**看到按频次取前 k 个，先用哈希计数，再按频次建立桶。频次最高不超过 n，从高频桶向低频收集，达到 k 个停止。

**易错点：**

- 同频元素顺序不要求

- 桶下标是频次

- 答案恰好 k 个

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
from typing import List
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = Counter(nums)
        return [x for x, _ in cnt.most_common(k)]
```

#### 295\. 数据流的中位数｜思路详解与标准代码

* [ ] ⭐⭐⭐ **295\. 数据流的中位数**：大根堆存较小一半、小根堆存较大一半；平衡使长度差不超过 1。

**思路：**看到数据流动态加入并随时取中位数，用两个堆维护有序分区。大根堆保存较小一半、小根堆保存较大一半，始终保证两边大小差不超过 1 且大根堆元素不大于小根堆元素。

**易错点：**

- Python 大根堆用负数模拟

- 每次加入后必须重平衡

- 偶数个元素取两堆顶均值

**复杂度：**addNum 时间 O\(log n\)，findMedian 时间 O\(1\)，额外空间 O\(n\)

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lower = []
        self.upper = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lower, -num)
        heapq.heappush(self.upper, -heapq.heappop(self.lower))
        if len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def findMedian(self) -> float:
        if len(self.lower) > len(self.upper):
            return float(-self.lower[0])
        return (-self.lower[0] + self.upper[0]) / 2.0
```

### 十四、贪心（4）

#### 121\. 买卖股票的最佳时机｜思路详解与标准代码

* [ ] ⭐⭐ **121\. 买卖股票的最佳时机**：维护历史最低价；当天卖出的收益 `price-minPrice` 更新答案。

**思路：**看到只能先买后卖一次，扫描时维护此前最低价格。以当天作为卖出日计算 price\-min\_price 更新答案，再更新最低价。

**易错点：**

- 不能在卖出日之后买入

- 无利润时返回 0

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List
from math import inf

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans, mi = 0, inf
        for v in prices:
            ans = max(ans, v - mi)
            mi = min(mi, v)
        return ans
```

#### 55\. 跳跃游戏｜思路详解与标准代码

* [ ] ⭐⭐⭐ **55\. 跳跃游戏**：维护最远可达 `far`；若 `i>far` 立即失败。

**思路：**看到能否到达末尾，贪心维护已知最远可达下标 far。扫描到 i 时若 i\>far 说明出现断点，否则用 i\+nums\[i\] 扩展 far。

**易错点：**

- 先判断当前位置是否可达

- far 到达 n\-1 可提前返回

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        mx = 0
        for i, x in enumerate(nums):
            if mx < i:
                return False
            mx = max(mx, i + x)
        return True
```

#### 45\. 跳跃游戏 II｜思路详解与标准代码

* [ ] ⭐⭐⭐ **45\. 跳跃游戏 II**：在当前层边界内更新下一层最远点；走到当前边界时步数 \+1 并换边界。

**思路：**看到保证可达且求最少跳数，把当前可达区间视为 BFS 一层。扫描当前层时维护下一层最远位置，走到当前层右边界就增加一步并更新边界。

**易错点：**

- 只扫描到 n\-2，避免终点再加一步

- 题目保证可达

- next\_end 在层内取最大值

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        ans = mx = last = 0
        for i, x in enumerate(nums[:-1]):
            mx = max(mx, i + x)
            if last == i:
                ans += 1
                last = mx
        return ans
```

#### 763\. 划分字母区间｜思路详解与标准代码

* [ ] ⭐⭐ **763\. 划分字母区间**：预存每字符最后位置；扫描扩展当前右界，到达右界就切一段。

**思路：**看到同一字符只能出现在一个片段，先记录每个字符最后位置。扫描当前片段时不断扩展 right 到所见字符的最远末位，i 到达 right 时切分。

**易错点：**

- 片段长度是 right\-start\+1

- 切分后 start 更新为 i\+1

**复杂度：**时间 O\(n\)，额外空间 O\(字符集大小\)

```python
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {c: i for i, c in enumerate(s)}
        mx = j = 0
        ans = []
        for i, c in enumerate(s):
            mx = max(mx, last[c])
            if mx == i:
                ans.append(i - j + 1)
                j = i + 1
        return ans
```

### 十五、动态规划（10）

#### 70\. 爬楼梯｜思路详解与标准代码

* [ ] ⭐ **70\. 爬楼梯**：`dp[i]=dp[i-1]+dp[i-2]`；滚动变量即可。

**思路：**看到每次走 1 或 2 阶的方案数，状态等同斐波那契递推。滚动维护到前两阶的方案数，迭代到 n。

**易错点：**

- n=1 时返回 1

- 滚动变量含义要一致

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return b
```

#### 118\. 杨辉三角｜思路详解与标准代码

* [ ] ⭐ **118\. 杨辉三角**：每行两端为 1，中间由上一行相邻两数之和得到。

**思路：**看到生成前 numRows 行杨辉三角，逐行构造。每行两端为 1，中间位置由上一行相邻两个数相加得到。

**易错点：**

- numRows=1 正确

- 每行长度为行号加一

**复杂度：**时间 O\(numRows²\)，额外空间 O\(numRows²\)（输出）

```python
from typing import List

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        triangle = []
        for row_index in range(numRows):
            row = [1] * (row_index + 1)
            for col in range(1, row_index):
                row[col] = triangle[-1][col - 1] + triangle[-1][col]
            triangle.append(row)
        return triangle
```

#### 198\. 打家劫舍｜思路详解与标准代码

* [ ] ⭐⭐⭐ **198\. 打家劫舍**：`dp[i]=max(dp[i-1],dp[i-2]+nums[i])`；“不偷/偷当前”二选一。

**思路：**看到相邻元素不能同时选择，滚动 DP 维护前一位置最优和前两位置最优。当前最优取“不偷当前”和“偷当前加前两位最优”的较大值。

**易错点：**

- 空数组返回 0

- 更新滚动变量时不能覆盖旧的前一状态

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        two_back = 0
        one_back = 0
        for value in nums:
            current = max(one_back, two_back + value)
            two_back, one_back = one_back, current
        return one_back
```

#### 279\. 完全平方数｜思路详解与标准代码

* [ ] ⭐⭐ **279\. 完全平方数**：完全背包/最短路；`dp[j]=min(dp[j],dp[j-square]+1)`。

**思路：**看到用最少完全平方数凑 n，使用完全背包 DP。dp\[x\] 表示凑成 x 的最少个数，对每个 x 枚举不超过它的平方数并取 dp\[x\-square\]\+1 最小值。

**易错点：**

- dp\[0\]=0

- 平方数可重复使用

- n 为正整数

**复杂度：**时间 O\(n√n\)，额外空间 O\(n\)

```python
from math import inf, sqrt

class Solution:
    def numSquares(self, n: int) -> int:
        m = int(sqrt(n))
        f = [[inf] * (n + 1) for _ in range(m + 1)]
        f[0][0] = 0
        for i in range(1, m + 1):
            for j in range(n + 1):
                f[i][j] = f[i - 1][j]
                if j >= i * i:
                    f[i][j] = min(f[i][j], f[i][j - i * i] + 1)
        return f[m][n]
```

#### 322\. 零钱兑换｜思路详解与标准代码

* [ ] ⭐⭐⭐ **322\. 零钱兑换**：完全背包求最少个数；初始化无穷，`dp[0]=0`，硬币可重复。

**思路：**看到硬币可重复且求最少数量，使用完全背包 DP。dp\[x\] 表示组成金额 x 的最少硬币数，从 dp\[0\]=0 向上转移，无穷仍保留表示不可达。

**易错点：**

- amount=0 返回 0

- 不可达返回 \-1

- 不要把组合数递推混入最少值问题

**复杂度：**时间 O\(amount·硬币数\)，额外空间 O\(amount\)

```python
from typing import List
from math import inf

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        m, n = len(coins), amount
        f = [[inf] * (n + 1) for _ in range(m + 1)]
        f[0][0] = 0
        for i, x in enumerate(coins, 1):
            for j in range(n + 1):
                f[i][j] = f[i - 1][j]
                if j >= x:
                    f[i][j] = min(f[i][j], f[i][j - x] + 1)
        return -1 if f[m][n] >= inf else f[m][n]
```

#### 139\. 单词拆分｜思路详解与标准代码

* [ ] ⭐⭐ **139\. 单词拆分**：`dp[i]` 表示前 `i` 字符可拆；枚举断点 `j`，要求 `dp[j]` 且 `s[j:i]` 在集合。

**思路：**看到字符串能否由词典词连续拼接，定义 dp\[i\] 为前 i 个字符可拆分。枚举结尾 i 和断点 j，只有 dp\[j\] 为真且 s\[j:i\] 在词典时才能令 dp\[i\] 为真。

**易错点：**

- dp\[0\]=True

- 词典转集合

- 可按最大词长减少无效枚举

**复杂度：**时间 O\(n²\)（切片成本视实现而定），额外空间 O\(n\+词典大小\)

```python
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        n = len(s)
        f = [True] + [False] * n
        for i in range(1, n + 1):
            f[i] = any(f[j] and s[j:i] in words for j in range(i))
        return f[n]
```

#### 300\. 最长递增子序列｜思路详解与标准代码

* [ ] ⭐⭐⭐ **300\. 最长递增子序列**：`tails` 维护各长度最小结尾；对每个数做 `lower_bound` 替换。$O(n\log n)$。

**思路：**看到严格递增子序列长度，维护 tails\[len\-1\] 为该长度的最小结尾。对每个数用 lower\_bound 找第一个大于等于它的位置替换，若不存在则追加。

**易错点：**

- 严格递增必须用 lower\_bound

- tails 本身不一定是原数组子序列

- 空数组返回 0

**复杂度：**时间 O\(n log n\)，额外空间 O\(n\)

```python
from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)
```

#### 152\. 乘积最大子数组｜思路详解与标准代码

* [ ] ⭐⭐⭐ **152\. 乘积最大子数组**：同时维护以当前结尾的最大/最小乘积；负数会交换二者角色。

**思路：**看到连续乘积且存在负数，必须同时维护以当前位置结尾的最大值与最小值。当前数为负时二者角色交换，再分别决定从当前重启或接续，更新全局最大值。

**易错点：**

- 负数会让最大最小互换

- 0 会自然重启

- 答案不能初始化为 0 以免全负单元素出错

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = f = g = nums[0]
        for x in nums[1:]:
            ff, gg = f, g
            f = max(x, ff * x, gg * x)
            g = min(x, ff * x, gg * x)
            ans = max(ans, f)
        return ans
```

#### 416\. 分割等和子集｜思路详解与标准代码

* [ ] ⭐⭐⭐ **416\. 分割等和子集**：总和为奇数直接否；转为容量 `sum/2` 的 0\-1 背包，容量倒序。

**思路：**看到能否分成和相等的两组，先把问题转成选若干数凑总和一半。总和为奇数直接失败；用 0\-1 背包布尔 DP，并让容量倒序更新避免重复使用元素。

**易错点：**

- 总和奇数立即返回 False

- 容量必须倒序

- dp\[0\]=True

**复杂度：**时间 O\(n·sum\)，额外空间 O\(sum\)

```python
from typing import List

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        m, mod = divmod(sum(nums), 2)
        if mod:
            return False
        n = len(nums)
        f = [[False] * (m + 1) for _ in range(n + 1)]
        f[0][0] = True
        for i, x in enumerate(nums, 1):
            for j in range(m + 1):
                f[i][j] = f[i - 1][j] or (j >= x and f[i - 1][j - x])
        return f[n][m]
```

#### 32\. 最长有效括号｜思路详解与标准代码

* [ ] ⭐⭐⭐ **32\. 最长有效括号**：DP：以 `i` 结尾，若 `s[i]=')'`，定位可能匹配的左括号并接上前段。

**思路：**看到最长连续合法括号，定义 dp\[i\] 为以 i 结尾的最长有效长度。仅当 s\[i\] 为右括号时，令 j=i\-dp\[i\-1\]\-1 定位可能配对的左括号；匹配后接上内部段和 j 前面的有效段。

**易错点：**

- j 必须先判非负

- 前段使用 dp\[j\-1\] 且 j=0 时为 0

- 答案取所有 dp\[i\] 最大值

**复杂度：**时间 O\(n\)，额外空间 O\(n\)

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        dp = [0] * n
        answer = 0
        for i in range(1, n):
            if s[i] == ')':
                j = i - dp[i - 1] - 1
                if j >= 0 and s[j] == '(':
                    dp[i] = dp[i - 1] + 2
                    if j > 0:
                        dp[i] += dp[j - 1]
                    answer = max(answer, dp[i])
        return answer
```

### 十六、多维动态规划（5）

#### 62\. 不同路径｜思路详解与标准代码

* [ ] ⭐⭐ **62\. 不同路径**：`dp[i][j]=上+左`；可压成一维，障碍不存在时也可组合数。

**思路：**看到只能向右或向下走的路径计数，网格 DP 的每格来自上方与左方。用一维 dp 压缩，初始化为 1，逐行令 dp\[j\]\+=dp\[j\-1\]。

**易错点：**

- 首行首列都只有一种走法

- m、n 均为正

**复杂度：**时间 O\(mn\)，额外空间 O\(n\)

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        f = [[0] * n for _ in range(m)]
        f[0][0] = 1
        for i in range(m):
            for j in range(n):
                if i:
                    f[i][j] += f[i - 1][j]
                if j:
                    f[i][j] += f[i][j - 1]
        return f[-1][-1]
```

#### 64\. 最小路径和｜思路详解与标准代码

* [ ] ⭐⭐ **64\. 最小路径和**：`dp[i][j]=grid[i][j]+min(上,左)`；首行首列单独初始化。

**思路：**看到只能右移或下移且求最小代价，状态为到达每格的最小路径和。可原地累加：首行首列只有单一来源，其余格加上上方与左方较小值。

**易错点：**

- 首行首列需单独处理

- 原地方案会修改 grid

**复杂度：**时间 O\(mn\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        for row in range(1, rows):
            grid[row][0] += grid[row - 1][0]
        for col in range(1, cols):
            grid[0][col] += grid[0][col - 1]
        for row in range(1, rows):
            for col in range(1, cols):
                grid[row][col] += min(grid[row - 1][col], grid[row][col - 1])
        return grid[-1][-1]
```

#### 5\. 最长回文子串｜思路详解与标准代码

* [ ] ⭐⭐ **5\. 最长回文子串**：中心扩展枚举奇/偶中心；或区间 DP，记录最长边界。

**思路：**看到最长连续回文子串，枚举每个奇数与偶数中心向两侧扩展。字符相等时继续扩大并更新最长边界，任一中心总扩展次数受字符串长度限制。

**易错点：**

- 必须同时处理奇偶中心

- 空串返回空串

- 相同最长长度返回任一均可

**复杂度：**时间 O\(n²\)，额外空间 O\(1\)

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        f = [[True] * n for _ in range(n)]
        k, mx = 0, 1
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                f[i][j] = False
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1]
                    if f[i][j] and mx < j - i + 1:
                        k, mx = i, j - i + 1
        return s[k : k + mx]
```

#### 1143\. 最长公共子序列｜思路详解与标准代码

* [ ] ⭐⭐⭐ **1143\. 最长公共子序列**：字符相等取左上 \+1，否则取上/左最大；明确是子序列非子串。

**思路：**看到两个字符串的最长子序列，DP 比较前缀。字符相等时取左上状态加一，否则取去掉任一末字符后的较大值；可用一维数组压缩。

**易错点：**

- 子序列不要求连续

- 一维压缩时要保存左上旧值

**复杂度：**时间 O\(mn\)，额外空间 O\(n\)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    f[i][j] = f[i - 1][j - 1] + 1
                else:
                    f[i][j] = max(f[i - 1][j], f[i][j - 1])
        return f[m][n]
```

#### 72\. 编辑距离｜思路详解与标准代码

* [ ] ⭐⭐⭐ **72\. 编辑距离**：`dp[i][j]` 为前缀距离；相同取左上，否则 `1+min(删、增、改)`。

**思路：**看到插入、删除、替换的最少次数，定义 dp\[i\]\[j\] 为两个前缀的编辑距离。末字符相同取左上，否则在删除、插入、替换三个前驱中取最小值加一。

**易错点：**

- 首行首列分别初始化为前缀长度

- 三种操作对应的前驱不能混淆

**复杂度：**时间 O\(mn\)，额外空间 O\(mn\)

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for j in range(1, n + 1):
            f[0][j] = j
        for i, a in enumerate(word1, 1):
            f[i][0] = i
            for j, b in enumerate(word2, 1):
                if a == b:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
        return f[m][n]
```

### 十七、技巧（5）

#### 136\. 只出现一次的数字｜思路详解与标准代码

* [ ] ⭐ **136\. 只出现一次的数字**：所有数异或；相同数抵消，`0^x=x`。

**思路：**看到其余元素都出现两次，利用异或的交换律和 x^x=0。把全部元素异或，成对元素抵消后即得到唯一元素。

**易错点：**

- 初值为 0

- 负数同样适用

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        answer = 0
        for value in nums:
            answer ^= value
        return answer
```

#### 169\. 多数元素｜思路详解与标准代码

* [ ] ⭐⭐ **169\. 多数元素**：Boyer–Moore 投票；相同加一、不同抵消，多数元素最终留下。

**思路：**看到某元素出现次数超过一半，使用 Boyer–Moore 投票。计数归零时更换候选，相同加一、不同减一；多数元素与其他元素抵消后仍会留下。

**易错点：**

- 题目保证多数元素存在

- 若不保证需第二遍验证

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cnt = m = 0
        for x in nums:
            if cnt == 0:
                m, cnt = x, 1
            else:
                cnt += 1 if m == x else -1
        return m
```

#### 75\. 颜色分类｜思路详解与标准代码

* [ ] ⭐⭐⭐ **75\. 颜色分类**：荷兰国旗；`[0,l)` 是 0，`[l,i)` 是 1，`(r,n)` 是 2；换来 2 时 `i` 不前进。

**思路：**看到只含 0、1、2 且要求原地一次遍历，使用荷兰国旗三指针。维护 \[0,left\) 为 0、\[left,i\) 为 1、\(right,n\) 为 2；遇 0 左换并前进，遇 2 右换但 i 不动。

**易错点：**

- 换入当前位置的右侧元素尚未检查

- 边界采用 left\<=right 或 i\<=right 要一致

- 原地修改

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        i, j, k = -1, len(nums), 0
        while k < j:
            if nums[k] == 0:
                i += 1
                nums[i], nums[k] = nums[k], nums[i]
                k += 1
            elif nums[k] == 2:
                j -= 1
                nums[j], nums[k] = nums[k], nums[j]
            else:
                k += 1
```

#### 31\. 下一个排列｜思路详解与标准代码

* [ ] ⭐⭐⭐ **31\. 下一个排列**：从右找首个下降位 `i`；从右找首个大于它的数交换；反转后缀为最小。

**思路：**看到求字典序紧邻的更大排列，从右找首个 nums\[i\]\<nums\[i\+1\] 的枢轴。再从右找首个大于枢轴的元素交换，并反转原本非递增的后缀使其最小；若无枢轴则整体反转。

**易错点：**

- 必须从右找严格大于枢轴的元素

- 后缀用反转即可

- 完全降序时回到最小排列

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        n = len(nums)
        i = next((i for i in range(n - 2, -1, -1) if nums[i] < nums[i + 1]), -1)
        if ~i:
            j = next((j for j in range(n - 1, i, -1) if nums[j] > nums[i]))
            nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1 :] = nums[i + 1 :][::-1]
```

#### 287\. 寻找重复数｜思路详解与标准代码

* [ ] ⭐⭐ **287\. 寻找重复数**：把 `i→nums[i]` 看成链表，重复值是环入口；Floyd 快慢指针。

**思路：**看到 n\+1 个值落在 \[1,n\] 且不能改数组，把 i→nums\[i\] 视为带环链表。Floyd 第一阶段让快慢指针相遇，第二阶段从起点与相遇点同速前进，环入口值就是重复数。

**易错点：**

- 起点使用 nums\[0\] 或 0 的模板要前后一致

- 不能修改原数组

- 重复值可能出现多次

**复杂度：**时间 O\(n\)，额外空间 O\(1\)

```python
from typing import List
from bisect import bisect_left

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        def f(x: int) -> bool:
            return sum(v <= x for v in nums) > x

        return bisect_left(range(len(nums)), True, key=f)
```

---

## 最后 60 分钟：只默写这 24 题

1. 3 无重复字符（滑窗）

2. 15 三数之和（排序双指针）

3. 42 接雨水（左右最大值）

4. 76 最小覆盖子串（滑窗计数）

5. 239 窗口最大值（单调队列）

6. 560 和为 K（前缀和哈希）

7. 41 缺失正数（原地哈希）

8. 206 反转链表

9. 142 环入口

10. 25 K 组反转

11. 148 链表归并排序

12. 146 LRU

13. 105 构造二叉树

14. 124 最大路径和

15. 437 树上前缀和

16. 207 拓扑排序

17. 46 全排列

18. 79 单词搜索

19. 34 二分左右边界

20. 84 柱状图最大矩形

21. 295 数据流中位数

22. 300 LIS

23. 416 0\-1 背包

24. 72 编辑距离

## 闭卷验收标准

对每道题依次回答下面 5 句，答不出就把复选框留空：

1. **识别信号**：题目哪个词让我想到这个算法？

2. **状态/不变量**：指针、窗口、栈、DFS 返回值或 DP 状态代表什么？

3. **一次迭代**：每轮具体更新什么，顺序为什么不能换？

4. **边界坑**：空输入、重复值、越界、初始化、恢复现场，最可能错哪一个？

5. **复杂度**：时间与额外空间是多少？

如果一题能在 **10 秒内说出算法，30 秒内说出关键不变量，3 分钟内写出骨架**，就算“被提醒能想起来”。

