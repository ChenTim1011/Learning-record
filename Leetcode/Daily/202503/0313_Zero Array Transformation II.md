[Zero Array Transformation II](https://leetcode.com/problems/zero-array-transformation-ii/description/?envType=daily-question&envId=2025-03-13)

## **📌 Problem Statement**
We are given:
1. An integer array `nums` of length `n`.
2. A 2D array `queries`, where each query is of the form `[li, ri, vali]`. 

Each query modifies `nums`:
- **Decrement each element in range `[li, ri]` by at most `vali`**.
- The decrement value can be chosen **independently** for each index.

Our goal:
- Find the **minimum k** such that after applying the **first k queries**, `nums` becomes **a Zero Array** (all elements `0`).
- If it's **impossible**, return `-1`.

---

## **🔹 Example Walkthrough**
### **Example 1**
#### **Input**
```cpp
nums = [2,0,2]
queries = [[0,2,1], [0,2,1], [1,1,3]]
```

#### **Step-by-step Execution**
1. **Query 0**: `[0,2,1]`  
   - We can decrement `nums[0]` by `1`, `nums[1]` by `0`, and `nums[2]` by `1`.
   - Result: `[1, 0, 1]`
  
2. **Query 1**: `[0,2,1]`  
   - We can decrement `nums[0]` by `1`, `nums[1]` by `0`, and `nums[2]` by `1`.
   - Result: `[0, 0, 0]` ✅ **Zero Array!**
   
**Output:** `2` (stopped at `k = 2` queries)

---

### **Example 2**
#### **Input**
```cpp
nums = [4,3,2,1]
queries = [[1,3,2], [0,2,1]]
```
#### **Step-by-step Execution**
1. **Query 0**: `[1,3,2]`  
   - We can decrement `nums[1]` by `2`, `nums[2]` by `2`, and `nums[3]` by `1`.
   - Result: `[4, 1, 0, 0]`

2. **Query 1**: `[0,2,1]`  
   - We can decrement `nums[0]` by `1`, `nums[1]` by `1`, `nums[2]` by `0`.
   - Result: `[3, 0, 0, 0]` ❌ **Not a Zero Array!**

**Output:** `-1` (impossible to reach a Zero Array)

---

## **🚀 Approach**
### **🔹 Key Observations**
1. **We need to check if `nums` can be zeroed using the first `k` queries.**
2. **Queries decrement values independently** → **Greedy approach** is useful.
3. **Efficient range updates** are required → **Difference Array** is useful.

---

### **🔹 Efficient Approach using Difference Array**
1. **Use a difference array** to efficiently apply queries.
   - Instead of modifying `nums` directly, we maintain a **diff array** to mark range updates.
   - Convert it back to `nums` in `O(n)` time.

2. **Binary Search on `k`**
   - **Check if `nums` can be reduced to zero** with first `k` queries.
   - If `k` queries are insufficient, try `k + 1`.
   - Use **binary search** to find the **minimum valid k**.

---

## **📝 Implementation**
```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minZeroArray(vector<int>& nums, vector<vector<int>>& queries) {
        int n = nums.size();
        int m = queries.size();
        
        // Binary search on the number of queries used
        int left = 0, right = m, ans = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            // Check if we can make nums zero using first `mid` queries
            if (canZero(nums, queries, mid)) {
                ans = mid;
                right = mid - 1;  // Try to minimize k
            } else {
                left = mid + 1;
            }
        }
        
        return ans;
    }

private:
    bool canZero(vector<int>& nums, vector<vector<int>>& queries, int k) {
        int n = nums.size();
        vector<int> diff(n + 1, 0);  // Difference array for range updates
        
        // Apply first `k` queries using a difference array
        for (int i = 0; i < k; i++) {
            int l = queries[i][0], r = queries[i][1], val = queries[i][2];
            diff[l] -= val;
            diff[r + 1] += val;
        }
        
        // Convert the difference array back to the actual modified array
        vector<int> modified(nums.begin(), nums.end());
        int current = 0;
        
        for (int i = 0; i < n; i++) {
            current += diff[i];  // Apply difference updates
            modified[i] += current;
            if (modified[i] > 0) return false;  // If any element remains > 0, `k` is insufficient
        }
        
        return true;  // If all elements are zero or negative, `k` is sufficient
    }
};
```

---

## **⏳ Complexity Analysis**
| **Operation**   | **Time Complexity** | **Explanation** |
|---------------|------------------|----------------|
| **Binary Search on `k`** | `O(log m)` | Since we are searching over `m` queries. |
| **Applying `k` Queries** | `O(n + k)` | Using **difference array** for range updates. |
| **Checking if `nums` is zero** | `O(n)` | Traversing the final array once. |
| **Total Complexity** | `O(n log m + k log m)` | Efficient for large constraints. |

---

## **✅ Summary**
- **Difference Array** optimizes range updates.
- **Binary Search on `k`** finds the **minimum valid k** efficiently.
- **Time Complexity:** `O(n log m)`, which is optimal for `n, m ≤ 10^5`.