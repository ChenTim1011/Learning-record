[Maximum Absolute Sum of Any Subarray](https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/description/)


## **Problem Statement**
You are given an integer array `nums`. The **absolute sum** of a subarray \([nums_l, nums_{l+1}, ..., nums_r]\) is defined as:  

\[
\text{abs}(\text{sum of subarray})
\]

where `abs(x)` is:  
- If \( x \geq 0 \), then \( \text{abs}(x) = x \).  
- If \( x < 0 \), then \( \text{abs}(x) = -x \).  

Your task is to **return the maximum absolute sum of any (possibly empty) subarray** of `nums`.

---

## **Example Walkthrough**

### **Example 1**
#### **Input:**
```cpp
nums = [1, -3, 2, 3, -4]
```
#### **Valid subarrays and their absolute sums:**
- `[2,3]` → sum = `2+3 = 5` → **abs(5) = 5**
- `[-3,2,3]` → sum = `-3+2+3 = 2` → **abs(2) = 2**
- `[1,-3,2,3]` → sum = `1-3+2+3 = 3` → **abs(3) = 3**
- `[-4]` → sum = `-4` → **abs(-4) = 4**

✅ The maximum absolute sum is `5`, so the output is:
```cpp
Output: 5
```

---

### **Example 2**
#### **Input:**
```cpp
nums = [2, -5, 1, -4, 3, -2]
```
#### **Valid subarrays and their absolute sums:**
- `[-5,1,-4]` → sum = `-5+1-4 = -8` → **abs(-8) = 8**
- `[3,-2]` → sum = `3-2 = 1` → **abs(1) = 1**
- `[2,-5,1]` → sum = `2-5+1 = -2` → **abs(-2) = 2**

✅ The maximum absolute sum is `8`, so the output is:
```cpp
Output: 8
```

---

## **Solution 1: Using Prefix Sum + STL**
### **💡 Key Idea**
Instead of checking every subarray (which is inefficient), we use **prefix sum** to efficiently find the maximum and minimum sum values.

### **Approach**
1. **Compute the prefix sum** using `std::partial_sum()`, updating `nums` in-place.
2. **Find the minimum and maximum prefix sum** using `std::minmax_element()`.
3. **Compute the result** using:
   \[
   \max(\text{max_prefix_sum}, 0) - \min(0, \text{min_prefix_sum})
   \]

### **🔹 Code**
```cpp
#include <iostream>
#include <vector>
#include <numeric>  // for partial_sum
#include <algorithm> // for minmax_element

using namespace std;

class Solution {
public:
    int maxAbsoluteSum(vector<int>& nums) {
        partial_sum(nums.begin(), nums.end(), nums.begin()); // Compute prefix sum
        auto [m, M] = minmax_element(nums.begin(), nums.end()); // Find min and max prefix sum
        return max(*M, 0) - min(0, *m);
    }
};
```

---

### **🛠 Understanding STL Functions Used**
#### **1️⃣ `std::partial_sum` (Compute Prefix Sum)**
```cpp
partial_sum(nums.begin(), nums.end(), nums.begin());
```
- Updates `nums[i]` to store the sum of `nums[0]` to `nums[i]`.

##### **Example**
```cpp
vector<int> nums = {1, -3, 2, 3, -4};
partial_sum(nums.begin(), nums.end(), nums.begin());
```
✅ Now `nums` becomes:
```
[1, -2, 0, 3, -1]
```

---

#### **2️⃣ `std::minmax_element` (Find Min and Max)**
```cpp
auto [m, M] = minmax_element(nums.begin(), nums.end());
```
- Returns iterators to the **smallest (`m`) and largest (`M`) elements** in `nums`.

##### **Example**
```cpp
vector<int> nums = {1, -2, 0, 3, -1};
auto [m, M] = minmax_element(nums.begin(), nums.end());
cout << "min: " << *m << ", max: " << *M << endl;
```
✅ Output:
```
min: -2, max: 3
```

---

### **⏱ Complexity Analysis**
| Time Complexity  | Space Complexity |
|-----------------|----------------|
| \(O(n)\)        | \(O(1)\)       |

- **`partial_sum(nums.begin(), nums.end(), nums.begin())` → \(O(n)\)**
- **`minmax_element(nums.begin(), nums.end())` → \(O(n)\)**
- Uses **constant extra space** (`O(1)`) since it modifies `nums` in-place.

---

## **🔹 Solution 2: Optimized Approach (Kadane’s Algorithm)**
### **💡 Key Idea**
Instead of storing prefix sums, **track two values during iteration**:
1. **`maxSum`** → Maximum sum of any subarray ending at the current position.
2. **`minSum`** → Minimum sum of any subarray ending at the current position.

At each step:
- **If `maxSum` goes negative, reset it to `0`.**
- **If `minSum` goes positive, reset it to `0`.**
- **Keep track of the maximum absolute difference:**
  \[
  \text{maxAbsSum} = \max(\text{maxSum} - \text{minSum})
  \]

### **🔹 Code**
```cpp
class Solution {
public:
    int maxAbsoluteSum(vector<int>& nums) {
        int maxSum = 0, minSum = 0;
        int maxAbsSum = 0;
        for (int x : nums) {
            maxSum = max(0, maxSum + x);
            minSum = min(0, minSum + x);
            maxAbsSum = max(maxAbsSum, maxSum - minSum);
        }
        return maxAbsSum;
    }
};
```

---

### **⏱ Complexity Analysis**
| Time Complexity | Space Complexity |
|---------------|----------------|
| \(O(n)\)     | \(O(1)\)       |

✅ **Advantages**:
- Uses **single pass** (\(O(n)\) complexity).
- **Does NOT modify `nums`**.
- Faster execution than prefix sum approach.

---

## **🔹 Final Comparison**
| Approach | Time Complexity | Space Complexity | Modifies `nums`? | Performance |
|----------|---------------|----------------|---------------|------------|
| **Prefix Sum + minmax_element** | \(O(n)\) | \(O(1)\) | ✅ Yes | Good |
| **Kadane’s Algorithm (Best)** | \(O(n)\) | \(O(1)\) | ❌ No | **Fastest (0ms, beats 100%)** |

---

## **🚀 Conclusion**
- **If you want a clean STL-based solution**, use **Prefix Sum + `minmax_element()`**.
- **For the most efficient solution**, use **Kadane’s Algorithm**, which runs in **single pass** and **does not modify the input**.

