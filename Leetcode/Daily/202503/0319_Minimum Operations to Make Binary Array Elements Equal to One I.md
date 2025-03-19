[Minimum Operations to Make Binary Array Elements Equal to One I](https://leetcode.com/problems/minimum-operations-to-make-binary-array-elements-equal-to-one-i/description/?envType=daily-question&envId=2025-03-19)

# **3191. Minimum Operations to Make Binary Array Elements Equal to One I – Detailed Explanation & Tutorial**  

## **📌 Problem Statement**  
We are given a **binary array** `nums`, consisting only of `0`s and `1`s.  

We can perform the following operation any number of times:  
- **Select any 3 consecutive elements** and **flip** all of them (change `0` to `1` and `1` to `0`).  

Our goal is to determine the **minimum number of operations** needed to make **all elements in `nums` equal to `1`**.  

If it is **impossible**, return `-1`.  

---

## **🔹 Example Walkthrough**  

### **Example 1**  
#### **Input:**  
```cpp
nums = [0,1,1,1,0,0]
```
#### **Operations:**
1. Flip indices **0, 1, 2** → `nums = [1,0,0,1,0,0]`
2. Flip indices **1, 2, 3** → `nums = [1,1,1,0,0,0]`
3. Flip indices **3, 4, 5** → `nums = [1,1,1,1,1,1]`

✅ **All elements are `1`, so the answer is `3`.**  

#### **Output:**  
```cpp
3
```

---

### **Example 2**  
#### **Input:**  
```cpp
nums = [0,1,1,1]
```
#### **Analysis:**  
- The array length is **4**, so every operation must flip **3 elements** at a time.  
- Since **there is an isolated `0` at the beginning**, we **can never include it in any valid operation**.  
- **Impossible to make all elements `1`.**  

#### **Output:**  
```cpp
-1
```

---

## **🔹 Key Observations**  
1. **The key challenge** is handling **isolated zeros (`0`s)** that cannot be flipped because we must always flip **3 consecutive elements**.  
2. If a `0` exists in a position where **it cannot be included in a group of 3**, it is **impossible** to turn everything into `1`s.  
3. **A greedy approach** is effective:  
   - Always flip the leftmost `0` using a valid **3-element window**.
   - Minimize the number of flips.

---

## **🔹 Approach: Greedy Sliding Window**  
### **💡 Idea:**  
1. **Iterate through the array** and identify the leftmost `0`.  
2. **Whenever we find a `0`, flip it with the next two elements**.  
3. **Repeat until all elements become `1`** or it becomes impossible.  

### **🔹 Implementation**
```cpp
class Solution {
public:
    int minOperations(vector<int>& nums) {
        int n = nums.size();
        int ops = 0;

        for (int i = 0; i <= n - 3; i++) {
            if (nums[i] == 0) {
                // Flip the three elements nums[i], nums[i+1], nums[i+2]
                nums[i] ^= 1;
                nums[i+1] ^= 1;
                nums[i+2] ^= 1;
                ops++;
            }
        }

        // Check if there are any remaining 0s
        for (int i = 0; i < n; i++) {
            if (nums[i] == 0) return -1;
        }

        return ops;
    }
};
```

---

## **🔹 Complexity Analysis**
| Approach | Time Complexity | Space Complexity | Explanation |
|----------|---------------|----------------|-------------|
| **Greedy Sliding Window** | **O(n)** | **O(1)** | We iterate through the array once (O(n)), flipping elements as needed. |

---

## **🔹 Summary**
✅ **Key Idea**: Always **flip the leftmost `0`** using a **3-element window**, and if a `0` remains unflipped, return `-1`.  
✅ **Approach**: **Greedy + Sliding Window** (process `0`s as early as possible).  
✅ **Time Complexity**: **O(n)**, which is efficient for large inputs (up to `10^5`).  

