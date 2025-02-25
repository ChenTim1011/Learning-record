[Number of Sub-arrays With Odd Sum](https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/description/?envType=daily-question&envId=2025-02-25)


## **Problem Statement**
Given an array of integers `arr`, return the **number of subarrays with an odd sum**.

Since the answer can be **very large**, return it **modulo** \( 10^9 + 7 \).

### **Examples**

#### **Example 1**
```
Input: arr = [1,3,5]
Output: 4
Explanation: 
All subarrays are [[1], [1,3], [1,3,5], [3], [3,5], [5]].
All subarray sums are [1, 4, 9, 3, 8, 5].
Odd sums are [1, 9, 3, 5] → total count = 4.
```

#### **Example 2**
```
Input: arr = [2,4,6]
Output: 0
Explanation: 
All subarrays are [[2], [2,4], [2,4,6], [4], [4,6], [6]].
All subarray sums are [2, 6, 12, 4, 10, 6].
All sums are even → total count = 0.
```

#### **Example 3**
```
Input: arr = [1,2,3,4,5,6,7]
Output: 16
```

### **🔹 Constraints**
- \( 1 \leq arr.length \leq 10^5 \)
- \( 1 \leq arr[i] \leq 100 \)

---

## **Approach (Efficient O(n) Solution)**
Instead of **checking all subarrays (O(n²) or O(n³))**, we use a **prefix sum approach** with a **counting trick**.

### **Key Observations**
1. Let `prefix_sum` represent the sum of elements from index `0` to `i`.
2. **A subarray sum is odd if:**  
   - `prefix_sum[j] - prefix_sum[i]` is odd.  
   - This happens when `prefix_sum[j]` and `prefix_sum[i]` have **different parity** (one is even, one is odd).
3. **Using this, we only need to track how many times we have seen an even or odd prefix sum!**  
   - Let `cnt[0]` store the **number of even prefix sums** seen so far.
   - Let `cnt[1]` store the **number of odd prefix sums** seen so far.

### **Steps**
1. **Initialize**:
   - `cnt[0] = 1` (since the empty prefix sum `0` is even).
   - `cnt[1] = 0` (no odd prefix sum initially).
   - `sum_is_odd = 0` (tracks if current prefix sum is odd).
   - `ans = 0` (stores result).
2. **Iterate through `arr`**:
   - Update `sum_is_odd` using XOR: `sum_is_odd ^= (x & 1)`.
   - If `sum_is_odd` is **odd**, add `cnt[0]` (previous even sums).
   - If `sum_is_odd` is **even**, add `cnt[1]` (previous odd sums).
   - Update `cnt[sum_is_odd]` (increment count for the current parity).
3. **Return `ans % MOD`**.

---

## **Code**
```cpp
class Solution {
public:
    static int numOfSubarrays(vector<int>& arr) {
        const int mod = 1e9 + 7;
        bool sum_is_odd = 0;
        int cnt[2] = {1, 0}; // cnt[0] for even, cnt[1] for odd
        long long ans = 0;

        for(int x : arr) {
            sum_is_odd ^= (x & 1);  // Update parity of prefix sum
            ans += cnt[1 - sum_is_odd];  // Count subarrays with odd sum
            cnt[sum_is_odd]++;  // Update count of even/odd prefix sums
        }

        return ans % mod;
    }
};
```

---

## **Explanation with Example**
### **Example: `arr = [1, 2, 3, 4]`**
We will track the `prefix_sum` and `cnt` array as we iterate:

| `i` | `arr[i]` | `prefix_sum` | `sum_is_odd` (0=even, 1=odd) | `cnt[0]` (even) | `cnt[1]` (odd) | Odd Subarrays (`ans`) |
|----|----|----|----|----|----|----|
| Start | - | 0 | even (0) | 1 | 0 | 0 |
| 0 | 1 | 1 | odd (1) | 1 | 1 | `cnt[0] = 1` |
| 1 | 2 | 3 | odd (1) | 1 | 2 | `cnt[0] + cnt[1] = 1 + 1 = 2` |
| 2 | 3 | 6 | even (0) | 2 | 2 | `cnt[1] + cnt[1] = 2 + 2 = 4` |
| 3 | 4 | 10 | even (0) | 3 | 2 | `cnt[1] + cnt[1] = 4 + 2 = 6` |

Final `ans = 6`.

---

## **Why Does This Work?**
### **1️⃣ Counting Trick**
- We track **how many times each prefix sum has been even or odd**.
- If `prefix_sum[j]` is **odd**, count how many **even** sums came before it.
- If `prefix_sum[j]` is **even**, count how many **odd** sums came before it.

### **2️⃣ XOR Trick (`sum_is_odd ^= (x & 1)`)**
- `(x & 1)` extracts **the last bit of x** (0 for even, 1 for odd).
- XOR `^=` updates `sum_is_odd` based on whether `x` is odd or even.

### **3️⃣ Efficient O(n) Solution**
- Instead of **checking all subarrays (O(n²))**, we use a **single pass (O(n))**.
- Uses only **O(1) extra space**.

---

## **🔹 Complexity Analysis**
| Operation | Complexity |
|-----------|------------|
| Iterating through `arr` | **O(n)** |
| Space Complexity | **O(1)** |

**⚡ Final Complexity:** **O(n) time, O(1) space** → **Optimal Solution** 🎯

---

## **🔹 Summary**
- **Using prefix sum parity, we count odd subarrays in O(n) time.**
- **We avoid checking all subarrays (O(n²)) and instead count prefix sums smartly.**
- **This is a common trick in prefix sum problems!** 🚀

