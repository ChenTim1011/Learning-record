[Check if Array Is Sorted and Rotated](https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/description/)


## Problem Description

**Goal:**  
Determine if an array is a "rotated sorted array". In other words, the array is initially sorted in non-decreasing order (i.e., each element is less than or equal to the next) and then possibly rotated by moving some elements from the beginning of the array to the end.

**What is a rotated sorted array?**  
Imagine you have a sorted array, for example:  
```
[1, 2, 3, 4, 5]
```
If you rotate (or shift) the array at some pivot, you might end up with:  
```
[3, 4, 5, 1, 2]
```
This is still considered valid because if you “rotate” it back, you would get the original sorted array.

**Key Property:**  
A rotated sorted array will have **at most one “drop”**. A “drop” is defined as a point in the array where an element is greater than its following element (i.e., `nums[i] > nums[i+1]`).

---

## Intuition

1. **Sorted Array vs. Rotated Sorted Array:**  
   - A completely sorted (non-decreasing) array (e.g., `[1, 2, 3, 4]`) can be seen as a rotated sorted array that has been rotated 0 times.
   - When the array is rotated, the order is mostly sorted except for one place where the array “wraps around” from a high value back to a low value.

2. **Counting the “Drops”:**  
   - If you traverse the array and count how many times an element is greater than the next element, a properly rotated sorted array will have **at most one drop**.
   - If there is more than one drop, then the array cannot be obtained by simply rotating a sorted array.

3. **Checking the Endpoints:**  
   - Since the array is rotated, the “drop” might occur between the last element and the first element.  
   - One approach is to include a check between `nums[n-1]` and `nums[0]` or to incorporate it into our counting method. The provided code uses an initial check comparing the first and last elements.

---

## Step-by-Step Approach

### 1. Initialize Variables

- **`n`**: the length of the array.
- **`count`**: a counter to track the number of drops (i.e., points where `nums[i] > nums[i+1]`).

### 2. Handle the Edge Case (Head-Tail Check)

- **Why?**  
  In a rotated sorted array, the smallest element (which starts the sorted sequence) may appear after the largest element because of the rotation.
  
- **How?**  
  Compare the first element (`nums[0]`) and the last element (`nums[n-1]`).  
  - If `nums[0] < nums[n-1]`, it suggests that the natural wrap-around drop might occur at the connection between the last element and the first element.  
  - In the given solution, this check is used to increment the drop counter (`count`).

### 3. Iterate Through the Array

- Loop through the array comparing each pair of adjacent elements (`nums[i]` and `nums[i+1]`).
- **If** `nums[i] > nums[i+1]`, then it is a drop:
  - Increment `count` by 1.
- **If** at any point `count` becomes greater than 1:
  - Return `false` immediately because it violates the rotated sorted array condition.

### 4. Return the Result

- If after the full iteration the number of drops is **at most 1** (`count <= 1`), then the array qualifies as a rotated sorted array.
- Otherwise, it is not.

---

## Example Walk-through

Let’s go through an example to see how the logic works.

### Example 1:  
**Input:** `nums = [3, 4, 5, 1, 2]`

1. **Initialization:**
   - `count = 0`
   - `n = 5`

2. **Head-Tail Check:**
   - Compare `nums[0]` (3) with `nums[n-1]` (2).  
     Since `3 > 2`, we do **not** increment the count here.

3. **Iteration through adjacent pairs:**
   - Compare `3` and `4`: no drop (3 < 4) → `count = 0`.
   - Compare `4` and `5`: no drop (4 < 5) → `count = 0`.
   - Compare `5` and `1`: **drop found** (5 > 1) → increment `count` to 1.
   - Compare `1` and `2`: no drop (1 < 2) → `count` remains 1.

4. **Final Check:**
   - Since `count = 1` (which is ≤ 1), return `true`.

### Example 2:  
**Input:** `nums = [2, 1, 3, 4]`

1. **Initialization:**
   - `count = 0`
   - `n = 4`

2. **Head-Tail Check:**
   - Compare `nums[0]` (2) with `nums[n-1]` (4).  
     Since `2 < 4`, increment `count` to 1.

3. **Iteration:**
   - Compare `2` and `1`: **drop found** (2 > 1) → increment `count` to 2.
   - Now `count` is greater than 1, so return `false` immediately.

---

## Complexity Analysis

- **Time Complexity:** O(n)  
  We only perform a single pass through the array.

- **Space Complexity:** O(1)  
  We use only a few extra variables, regardless of the input size.

---

## Complete Code (C++)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool check(vector<int>& nums) {
        int n = nums.size();
        int count = 0;
        
        // Check the connection between the first and last element.
        if (nums[0] < nums[n - 1])
            count++;
        
        // Traverse through the array to count the "drop" points.
        for (int i = 0; i < n - 1; i++) {
            if (nums[i] > nums[i + 1])
                count++;
            if (count > 1)
                return false;
        }
        return true;
    }
};
```

---

## Summary

- **Problem Recap:**  
  Determine if the given array is obtained by taking a sorted array and rotating it, meaning there should be at most one point where the order "drops".

- **Key Insight:**  
  By counting the number of times an element is greater than the next one, we can verify if there is more than one drop. A count of 0 or 1 is acceptable.

- **Implementation Steps:**  
  1. Initialize a counter and get the array length.
  2. Optionally check the relation between the first and last elements.
  3. Iterate over adjacent pairs to count the drops.
  4. Return `true` if the count is at most 1; otherwise, return `false`.

