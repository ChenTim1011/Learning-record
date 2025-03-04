[Partition Array According to Given Pivot](https://leetcode.com/problems/partition-array-according-to-given-pivot/description/)

## **📌 Problem Statement**  
We are given an integer array `nums` and an integer `pivot`.  
We need to **rearrange `nums`** such that:  
1. **Elements < pivot** appear **first**.  
2. **Elements == pivot** appear **in the middle**.  
3. **Elements > pivot** appear **last**.  
4. The **relative order** of elements **< pivot** and **> pivot** is maintained.  

---

## **💡 Approach: Using Three Lists**  
Since we need to **preserve order**, we use three separate arrays:  
1. **`smaller`** → Stores numbers **less than pivot**.  
2. **`equal`** → Stores numbers **equal to pivot**.  
3. **`greater`** → Stores numbers **greater than pivot**.  

### **Steps**  
1. **Iterate through `nums`** and categorize each number into one of the three lists.  
2. **Concatenate** the lists in order: `[smaller] + [equal] + [greater]`.  
3. **Return the final list**.  

---

## **🚀 Optimized C++ Solution**  
```cpp
class Solution {
public:
    vector<int> pivotArray(vector<int>& nums, int pivot) {
        vector<int> smaller, equal, greater;
        
        // Categorize elements
        for (int num : nums) {
            if (num < pivot) smaller.push_back(num);
            else if (num == pivot) equal.push_back(num);
            else greater.push_back(num);
        }
        
        // Merge all three lists
        vector<int> result;
        result.insert(result.end(), smaller.begin(), smaller.end());
        result.insert(result.end(), equal.begin(), equal.end());
        result.insert(result.end(), greater.begin(), greater.end());
        
        return result;
    }
};
```

---

## **💡 Complexity Analysis**  
| Complexity  | Explanation |  
|------------|------------|  
| **Time Complexity** | **O(n)** → We traverse `nums` once and construct `result` in linear time. |  
| **Space Complexity** | **O(n)** → We store elements in three separate lists. |  

---

## **✅ Edge Cases Considered**  
### **1️⃣ Already Sorted Input**  
```cpp
nums = [1, 2, 3, 4, 5], pivot = 3
```
✔ Output: `[1, 2, 3, 4, 5]` (No changes)  

---

### **2️⃣ All Elements are the Pivot**  
```cpp
nums = [5, 5, 5, 5], pivot = 5
```
✔ Output: `[5, 5, 5, 5]` (No changes)  

---

### **3️⃣ Pivot is at the Start or End**  
```cpp
nums = [10, 15, 20, 10, 5, 1, 10], pivot = 10
```
✔ Output: `[5, 1, 10, 10, 10, 15, 20]` (Correctly partitions)  

---

### **4️⃣ Mixed Negative and Positive Numbers**  
```cpp
nums = [-5, -1, 0, 3, -2, 2, 1], pivot = 0
```
✔ Output: `[-5, -1, -2, 0, 3, 2, 1]`  

---

## **🔹 Summary**  
✅ **Three lists method ensures order is preserved.**  
✅ **Runs in `O(n)` time complexity.**  
✅ **Handles all edge cases effectively.**
