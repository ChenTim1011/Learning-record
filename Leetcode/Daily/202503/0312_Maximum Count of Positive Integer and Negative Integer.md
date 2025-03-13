[Maximum Count of Positive Integer and Negative Integer](https://leetcode.com/problems/maximum-count-of-positive-integer-and-negative-integer/description/?envType=daily-question&envId=2025-03-12)

## **📌 Problem Statement**
We are given a **sorted (non-decreasing) array** `nums`. Our goal is to **find the maximum** between:
- The count of **negative numbers**
- The count of **positive numbers**

**Note:** `0` is neither positive nor negative.

---

## **🔹 Example Walkthrough**
### **Example 1**
#### **Input:**
```cpp
nums = [-2,-1,-1,1,2,3]
```
#### **Count:**
- **Negatives:** `[-2, -1, -1]` → **3**
- **Positives:** `[1, 2, 3]` → **3**

#### **Output:**
```cpp
3
```

---

### **Example 2**
#### **Input:**
```cpp
nums = [-3,-2,-1,0,0,1,2]
```
#### **Count:**
- **Negatives:** `[-3, -2, -1]` → **3**
- **Positives:** `[1, 2]` → **2**

#### **Output:**
```cpp
3
```

---

### **Example 3**
#### **Input:**
```cpp
nums = [5, 20, 66, 1314]
```
#### **Count:**
- **Negatives:** `[]` → **0**
- **Positives:** `[5, 20, 66, 1314]` → **4**

#### **Output:**
```cpp
4
```

---

## **🚀 Approach**
### **🔹 Brute Force (O(n))**
1. Traverse the array.
2. Count **negative** and **positive** numbers.
3. Return the **maximum** count.

**Time Complexity:** `O(n)`

---

### **🔹 Optimized Approach (O(log n)) using Binary Search**
#### **Key Observations**
- Since the array is **sorted**, all negatives appear **before zero**, and all positives appear **after zero**.
- We can find:
  - The **first non-negative index** (start of `0` or `positive` numbers).
  - The **first positive index** (start of `positive` numbers).
- The count of **negatives** is the index of the first non-negative.
- The count of **positives** is the size of the array minus the index of the first positive.

---

## **📝 Binary Search Implementation**
```cpp
class Solution {
public:
    int maximumCount(vector<int>& nums) {
        int n = nums.size();

        // Find first non-negative index (smallest index where nums[i] >= 0)
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < 0)
                left = mid + 1;
            else
                right = mid - 1;
        }
        int negCount = left;  // First non-negative index is count of negatives

        // Find first positive index (smallest index where nums[i] > 0)
        left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= 0)
                left = mid + 1;
            else
                right = mid - 1;
        }
        int posCount = n - left;  // First positive index count

        return max(negCount, posCount);
    }
};
```

---

## **⏳ Complexity Analysis**
| **Approach** | **Time Complexity** | **Space Complexity** |
|-------------|--------------------|--------------------|
| **Brute Force (O(n))** | **O(n)** | **O(1)** |
| **Binary Search (O(log n))** | **O(log n)** | **O(1)** |

