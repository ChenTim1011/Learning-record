[Search Insert Position](https://leetcode.com/problems/search-insert-position/description/)


## My solution

### Code with Detailed English Comments

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        int result = 0;
        bool equal = false; // Flag to check if the target exists in the array

        // Binary search to locate the target or potential insertion point
        while (left < right) { 
            int mid = (left + right) / 2; // Calculate the mid-point
            if (nums[mid] == target) { // Target found
                result = mid;
                equal = true;
                break; // Exit the loop
            }
            if (nums[mid] < target) { // Target is in the right half
                left = mid + 1;
            }
            if (nums[mid] > target) { // Target is in the left half
                right = mid - 1;
            }
        }

        // If the target exists in the array
        if (equal == true) {
            return result;
        }

        // Handle the insertion position when the target does not exist
        // If the target is less than or equal to nums[left], return left
        if (target <= nums[left]) {
            return left;
        }
        // Otherwise, the target should be inserted after nums[left]
        return left + 1;
    }
};
```

---

### Explanation of Your Original Code

1. **Mistake 1: Missed Boundary Condition for Insertion**
   ```cpp
   if (target <= nums[left]) {
       return left;
   }
   return left + 1;
   ```
   - In your original code, you forgot to include these conditions, which caused the program to return incorrect results for cases where the target needed to be inserted either at the beginning of the array or at a position where `target > nums[left]`.
   - Without this check, the program cannot determine the correct insertion point when the target does not match an existing element.

   **Example Problematic Case:**
   ```cpp
   nums = [1, 3, 5, 6], target = 2
   ```
   Without the conditions, the program would fail to return the correct insertion index `1`.

2. **Mistake 2: Use of `while (left < right)` Causing TLE**
   ```cpp
   while (left < right)
   ```
   - This condition does not allow the loop to narrow down to a single valid index (`left == right`) when the search space reduces to one element.
   - As a result, the loop may fail to terminate in edge cases, leading to a **Time Limit Exceeded (TLE)** error.
   - **Fix:** Use `while (left <= right)` instead, which ensures that all valid indices are considered, including when `left == right`.

---

### Corrected and Optimized Code

Here’s an optimized version with proper boundary checks and adjustments to avoid TLE:

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;

        // Binary search to find the target or its insertion point
        while (left <= right) { // Use <= to include all valid indices
            int mid = left + (right - left) / 2; // Avoid overflow
            if (nums[mid] == target) { // Target found
                return mid;
            } else if (nums[mid] < target) { // Target is in the right half
                left = mid + 1;
            } else { // Target is in the left half
                right = mid - 1;
            }
        }

        // At the end of the loop, 'left' will be the correct insertion point
        return left;
    }
};
```

---

### Key Improvements

1. **Fixed Boundary Condition**:
   - The corrected code uses `while (left <= right)` to ensure the loop properly considers all valid indices, including edge cases.

2. **Simplified Insertion Logic**:
   - The binary search is modified so that `left` naturally becomes the insertion index when the loop ends. No need for extra checks like:
     ```cpp
     if (target <= nums[left]) { return left; }
     return left + 1;
     ```
   - This reduces redundancy and makes the code more elegant.

3. **Avoid TLE**:
   - Using `while (left < right)` in your original code could prevent the loop from terminating under certain conditions. The updated `while (left <= right)` ensures proper termination.

---

### Example Input/Output

#### Input:
```cpp
nums = [1, 3, 5, 6], target = 2
```

#### Output:
```cpp
1
```

#### Input:
```cpp
nums = [1, 3, 5, 6], target = 7
```

#### Output:
```cpp
4
```

#### Input:
```cpp
nums = [1, 3, 5, 6], target = 0
```

#### Output:
```cpp
0
```

