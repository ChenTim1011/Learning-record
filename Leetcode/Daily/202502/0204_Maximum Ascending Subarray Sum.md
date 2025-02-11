[Maximum Ascending Subarray Sum](https://leetcode.com/problems/maximum-ascending-subarray-sum/description/)

The runtime error in the original code occurs because of an **out-of-bounds memory access**. Let me explain the issue and how to fix it step by step.

---

### Problem Analysis
1. **Cause of the Issue**:
   - The loop condition in your code is `i + 1 <= nums.size()`. This ensures that `i + 1` does not exceed the size of the array.
   - However, inside the loop, you access `nums[i + 1]`. When `i == nums.size() - 1`, the condition `i + 1 <= nums.size()` holds true (since `nums.size() == i + 1`), but accessing `nums[i + 1]` attempts to read memory beyond the array's valid range. This causes a **heap-buffer-overflow** error.

2. **Specific Error**:
   - The problematic line is:
     ```cpp
     if (nums[i] >= nums[i + 1]) {
     ```
   - When `i + 1 == nums.size()`, the program tries to access `nums[nums.size()]`, which is invalid because the valid indices of the array are from `0` to `nums.size() - 1`.

---

### Solution
To fix the issue, we need to adjust the loop condition to ensure that the program does not attempt to access `nums[i + 1]` when `i` is at the last index.

---

### Fixed Code
```cpp
class Solution {
public:
    int maxAscendingSum(vector<int>& nums) {
        if (nums.size() == 1) return nums[0]; // Handle single-element case
        int sum = nums[0]; // Initialize sum to the first element
        int maxsum = nums[0]; // Initialize maxsum to the first element
        for (int i = 1; i < nums.size(); i++) { // Loop starts from index 1
            if (nums[i] > nums[i - 1]) { // Check if the sequence is ascending
                sum += nums[i]; // Add current element to sum
            } else {
                sum = nums[i]; // Reset sum to the current element
            }
            maxsum = max(maxsum, sum); // Update maxsum if sum is greater
        }
        return maxsum;
    }
};
```

---

### Explanation of Changes
1. **Loop Condition Adjustment**:
   - The loop now runs from `i = 1` to `i < nums.size()`. This ensures that the program never attempts to access `nums[i + 1]`.
   - Instead of comparing `nums[i]` with `nums[i + 1]`, we compare `nums[i]` with `nums[i - 1]`.

2. **Initialization**:
   - `sum` and `maxsum` are both initialized to the first element of the array, `nums[0]`. This avoids unnecessary checks for edge cases like a single-element array.

3. **Logic for Ascending Sequence**:
   - If `nums[i] > nums[i - 1]`, the current element is part of an ascending sequence, so it is added to the running `sum`.
   - Otherwise, the sequence ends, and `sum` is reset to the current element (`nums[i]`).

4. **Updating Maximum Sum**:
   - After processing each element, `maxsum` is updated with the greater value between the current `maxsum` and the running `sum`.

---

### Example Walkthrough
#### Input:
```cpp
nums = {10, 20, 30, 5, 10, 50};
```

#### Execution:
- **Step 1**: `sum = 10`, `maxsum = 10` (initialize)
- **Step 2**: `nums[1] > nums[0] (20 > 10) → sum = 30`, `maxsum = 30`
- **Step 3**: `nums[2] > nums[1] (30 > 20) → sum = 60`, `maxsum = 60`
- **Step 4**: `nums[3] <= nums[2] (5 <= 30) → sum = 5`, `maxsum = 60`
- **Step 5**: `nums[4] > nums[3] (10 > 5) → sum = 15`, `maxsum = 60`
- **Step 6**: `nums[5] > nums[4] (50 > 10) → sum = 65`, `maxsum = 65`

#### Output:
```
65
```

---

### Edge Cases
1. **Single Element**:
   - Input: `{100}`
   - Output: `100`

2. **Strictly Ascending**:
   - Input: `{1, 2, 3, 4, 5}`
   - Output: `15`

3. **Strictly Descending**:
   - Input: `{5, 4, 3, 2, 1}`
   - Output: `5`

4. **Mixed Sequence**:
   - Input: `{10, 20, 10, 40, 30, 50}`
   - Output: `80`

---

### Key Takeaways
- Always carefully check loop conditions to prevent out-of-bounds access.
- When accessing an array element based on an index (like `nums[i + 1]`), ensure the index is within bounds.
- AddressSanitizer is a powerful tool for detecting such issues and provides detailed error reports to pinpoint the exact cause.