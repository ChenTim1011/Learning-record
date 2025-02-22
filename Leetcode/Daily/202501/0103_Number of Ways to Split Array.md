[Number of Ways to Split Array](https://leetcode.com/problems/number-of-ways-to-split-array/description/?envType=daily-question&envId=2025-01-03)

## **Problem Breakdown**
We are given:
- An **array `nums` of length `n`**.
- We need to count **the number of valid splits** in `nums`.

A **valid split** at index `i` satisfies:
1. The sum of the first `i+1` elements **(left part)** is **greater than or equal** to the sum of the last `n-i-1` elements **(right part)**.
2. There must be **at least one element in the right part**, meaning `0 ≤ i < n-1`.

---

## **Understanding with an Example**
### **Example 1**
#### **Input**
```cpp
nums = [10,4,-8,7]
```
#### **Step 1: Compute Total Sum**
```
Total sum = 10 + 4 + (-8) + 7 = 13
```

#### **Step 2: Try Splitting at Each Index**
| Index `i` | Left Part | Sum (Left) | Right Part | Sum (Right) | Valid Split? |
|-----------|----------|------------|------------|------------|--------------|
| 0         | [10]     | 10         | [4,-8,7]   | 3          | ✅ Yes (10 ≥ 3) |
| 1         | [10,4]   | 14         | [-8,7]     | -1         | ✅ Yes (14 ≥ -1) |
| 2         | [10,4,-8]| 6          | [7]        | 7          | ❌ No (6 < 7) |

Thus, **valid splits occur at index 0 and 1**, and the answer is **`2`**.

#### **Output**
```cpp
2
```

---

## **Optimized Approach: Prefix Sum**
### **Why Use Prefix Sum?**
Instead of recomputing the sum of the left and right parts for every index, we use a **prefix sum array** to store cumulative sums. This allows us to compute left and right sums efficiently in **O(1) time** per query.

### **Steps to Implement**
### **Step 1️⃣: Compute Prefix Sum**
- `prefix[i]` stores **the sum of elements from index `0` to `i-1`**.

### **Step 2️⃣: Compute Total Sum**
- `total = prefix[n]` (sum of all elements in `nums`).

### **Step 3️⃣: Count Valid Splits**
- For each index `i` (`0 ≤ i < n-1`), check:
  \[
  \text{left sum} = \text{prefix}[i+1]
  \]
  \[
  \text{right sum} = \text{total} - \text{prefix}[i+1]
  \]
- If `left sum ≥ right sum`, increment the count.

---

## **Code Implementation**
### **C++ Code**
```cpp
class Solution {
public:
    int waysToSplitArray(vector<int>& nums) {
        int answer = 0;
        int n = nums.size();
        
        // Step 1: Compute Prefix Sum
        vector<long long> prefixsum(n + 1, 0);
        for (int i = 0; i < n; i++) {
            prefixsum[i + 1] = prefixsum[i] + nums[i];
        }

        // Step 2: Compute Total Sum
        long long total = prefixsum[n];

        // Step 3: Count Valid Splits
        for (int i = 0; i < n - 1; i++) { // We must leave at least one element on the right side
            long long left_sum = prefixsum[i + 1];
            long long right_sum = total - left_sum;
            if (left_sum >= right_sum) {
                answer++;
            }
        }
        
        return answer;
    }
};
```

---

## **Time & Space Complexity Analysis**
### **Time Complexity**
- **Step 1: Compute Prefix Sum** → `O(n)`
- **Step 2: Compute Total Sum** → `O(1)`
- **Step 3: Loop Through `n-1` Elements and Check Condition** → `O(n)`
- **Total Time Complexity: `O(n)`** ✅

### **Space Complexity**
- **Prefix sum array** → `O(n)`
- **Other variables** → `O(1)`
- **Total Space Complexity: `O(n)`** ✅ (Can be optimized to `O(1)`, explained below)

---

## **Optimized Approach: Using a Running Sum (`O(1)` Space)**
Instead of using an extra prefix sum array, we can maintain a **single variable** to store the left sum as we iterate.

### **Optimized Code (O(1) Space)**
```cpp
class Solution {
public:
    int waysToSplitArray(vector<int>& nums) {
        int answer = 0;
        long long left_sum = 0;
        long long total = accumulate(nums.begin(), nums.end(), 0LL);
        
        for (int i = 0; i < nums.size() - 1; i++) { // Ensure at least one element on the right
            left_sum += nums[i]; // Keep track of left sum
            long long right_sum = total - left_sum;
            if (left_sum >= right_sum) {
                answer++;
            }
        }
        
        return answer;
    }
};
```

### **Why is This More Efficient?**
✅ **Avoids using an extra array (`O(1)` space instead of `O(n)`)**  
✅ **Still runs in `O(n)` time complexity**  
✅ **Uses a single loop to keep track of sums efficiently**

---

## **Example Walkthrough (Optimized Approach)**
### **Input**
```cpp
nums = [10,4,-8,7]
```
### **Step 1: Compute Total Sum**
```
total = 10 + 4 + (-8) + 7 = 13
```
### **Step 2: Iterate Through Array and Maintain Running Sum**
| `i` | `nums[i]` | `left_sum` (Running) | `right_sum = total - left_sum` | Valid Split? |
|----|----|----|----|----|
| 0 | 10 | 10 | `13 - 10 = 3` | ✅ Yes (10 ≥ 3) |
| 1 | 4  | 14 | `13 - 14 = -1` | ✅ Yes (14 ≥ -1) |
| 2 | -8 | 6  | `13 - 6 = 7`  | ❌ No (6 < 7) |

### **Final Answer**
```cpp
2
```

---

## **Summary**
✅ **Using Prefix Sum allows `O(1)` query time for each split check**  
✅ **Final optimized solution uses `O(1)` space**  
✅ **Time Complexity: `O(n)`, Space Complexity: `O(1)`**  
✅ **Handles large constraints (`n ≤ 10^5`) efficiently**  

