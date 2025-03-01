[Apply Operations to an Array](https://leetcode.com/problems/apply-operations-to-an-array/description/)

## **📌 Problem Statement**
You are given a **0-indexed** array `nums` of size `n` consisting of **non-negative** integers.

🔹 You need to apply **(n - 1) operations**, where in the `i-th` operation:  
- If `nums[i] == nums[i + 1]`, then:
  - **Multiply `nums[i]` by `2`**.
  - **Set `nums[i + 1]` to `0`**.
- Otherwise, **skip this operation**.

🔹 **After applying all operations**, shift **all the `0`s** to the end of the array.

🔹 **Return the resulting array**.

---

## **Example Walkthrough**
### **Example 1**
#### **Input:**  
```cpp
nums = [1,2,2,1,1,0]
```
#### **Processing:**
1. `nums[1] == nums[2]` → **Merge** (`nums[1] *= 2, nums[2] = 0`) → `[1,4,0,1,1,0]`
2. `nums[3] == nums[4]` → **Merge** (`nums[3] *= 2, nums[4] = 0`) → `[1,4,0,2,0,0]`
3. **Shift all `0`s** to the end → `[1,4,2,0,0,0]`

#### **Output:**  
```cpp
[1,4,2,0,0,0]
```

---

### **Example 2**
#### **Input:**  
```cpp
nums = [0,1]
```
#### **Processing:**
- No **adjacent equal elements**, so no operations are applied.
- **Shift `0` to the end** → `[1,0]`.

#### **Output:**  
```cpp
[1,0]
```

---

## **Optimized Approach**
### **Two Steps**
1. **Merge Adjacent Equal Numbers** (Modify `nums` in-place).
2. **Move All `0`s to the End** (Preserve the order of nonzero numbers).

---

## **Optimized C++ Solution**
```cpp
class Solution {
public:
    vector<int> applyOperations(vector<int>& nums) {
        int n = nums.size();

        // Step 1: Merge adjacent equal numbers
        for (int i = 0; i < n - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
        }

        // Step 2: Shift all zeros to the end
        vector<int> result;
        for (int num : nums) {
            if (num != 0) result.push_back(num);
        }
        while (result.size() < n) {
            result.push_back(0);
        }

        return result;
    }
};
```

---

## **Explanation of Code**
### **1️⃣ Merge Adjacent Equal Numbers**
```cpp
for (int i = 0; i < n - 1; i++) {
    if (nums[i] == nums[i + 1]) {
        nums[i] *= 2;  // Double nums[i]
        nums[i + 1] = 0;  // Set nums[i+1] to 0
    }
}
```
- **If two adjacent elements are equal**, we **double** the first and **set the second to `0`**.

---

### **2️⃣ Move All `0`s to the End**
```cpp
vector<int> result;
for (int num : nums) {
    if (num != 0) result.push_back(num); // Store non-zero elements
}
while (result.size() < n) {
    result.push_back(0); // Fill remaining slots with zeros
}
```
- **Extract nonzero elements** into `result`.
- **Append `0`s to the end** to maintain the same length.

---

## **Complexity Analysis**
| Complexity | Explanation |
|------------|------------|
| **Time Complexity** | **O(n)** → We traverse `nums` twice (once for merging, once for shifting). |
| **Space Complexity** | **O(n)** → We use an extra array `result`. |

---

## **Space-Optimized Approach (In-Place)**
We can **avoid using extra space** and modify `nums` directly.

```cpp
class Solution {
public:
    vector<int> applyOperations(vector<int>& nums) {
        int n = nums.size();

        // Step 1: Merge adjacent equal numbers
        for (int i = 0; i < n - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
        }

        // Step 2: Shift nonzero elements forward
        int index = 0;  // Pointer for placing nonzero elements
        for (int i = 0; i < n; i++) {
            if (nums[i] != 0) {
                swap(nums[index], nums[i]);
                index++;
            }
        }

        return nums;
    }
};
```

---

## **Explanation of Space-Optimized Approach**
### **1️⃣ Merge Adjacent Equal Numbers**
(Same as before)

---

### **2️⃣ Shift Nonzero Elements (In-Place)**
```cpp
int index = 0;  
for (int i = 0; i < n; i++) {
    if (nums[i] != 0) {
        swap(nums[index], nums[i]); // Move nonzero element to correct position
        index++;
    }
}
```
- **Two-pointer technique** to move **nonzero elements forward**.
- **Swaps elements in-place**, reducing space complexity to **O(1)**.

---

## **Complexity Analysis (Optimized)**
| Complexity | Explanation |
|------------|------------|
| **Time Complexity** | **O(n)** → We traverse `nums` twice (once for merging, once for shifting). |
| **Space Complexity** | **O(1)** → We modify `nums` **in-place** without using extra space. |

---

## **Edge Cases Considered**
✅ **All elements are distinct**  
   - Example: `nums = [1,2,3,4]`  
   - Output: `[1,2,3,4]` (No change)

✅ **All elements are equal**  
   - Example: `nums = [2,2,2,2]`  
   - Output: `[4,4,0,0]`

✅ **Contains multiple `0`s**  
   - Example: `nums = [0,0,1,1,2,2]`  
   - Output: `[2,2,4,0,0,0]`

✅ **Already sorted output case**  
   - Example: `nums = [1,1,0,2,2,0,3]`  
   - Output: `[2,4,3,0,0,0,0]`

✅ **Large inputs handled efficiently**  
   - Handles `nums.length = 10^5` within **O(n) time**.

---

## **Summary**
✅ **O(n) time complexity, making it optimal**  
✅ **Space-efficient in-place solution (`O(1)`)**  
✅ **Handles all edge cases correctly**  
✅ **Uses two-pointer technique for efficiency**  
