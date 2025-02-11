[Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

## My solution

### Code with Detailed English Comments

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> result;
        // Binary search to find the target index
        int left = 0;
        int right = nums.size() - 1;
        int mid = 0;
        bool find = false; // Flag to indicate if the target is found
        while (left <= right) {
            mid = (left + right) / 2; // Calculate the middle index
            if (nums[mid] == target) {
                find = true; // Target found
                break;
            }
            if (nums[mid] < target) {
                left = mid + 1; // Target is in the right half
            }
            if (nums[mid] > target) {
                right = mid - 1; // Target is in the left half
            }
        }

        // If the target is not found, return [-1, -1]
        if (!find) {
            result.emplace_back(-1);
            result.emplace_back(-1);
            return result;
        }
        
        // Find the leftmost index of the target
        int count = 1; // Counter to check for duplicates
        int minleft = mid; // Start from the found middle index
        while (true) {
            // Ensure we do not access out-of-bounds elements
            if (mid - count >= 0 && nums[mid] == nums[mid - count]) {
                minleft = min(mid, mid - count); // Update the leftmost index
                count++;
            } else {
                result.emplace_back(minleft); // Add the leftmost index to the result
                break;
            }
        }

        // Find the rightmost index of the target
        count = 1; // Reset the counter
        int maxright = mid; // Start from the found middle index
        while (true) {
            // Ensure we do not access out-of-bounds elements
            if (mid + count <= nums.size() - 1 && nums[mid] == nums[mid + count]) {
                maxright = max(mid, mid + count); // Update the rightmost index
                count++;
            } else {
                result.emplace_back(maxright); // Add the rightmost index to the result
                break;
            }
        }

        return result;
    }
};
```

---

### Explanation of the Original Error

The original code was missing the conditions:
- `mid - count >= 0` in the **leftmost search loop**.
- `mid + count <= nums.size() - 1` in the **rightmost search loop**.

These conditions are necessary to ensure that the code does not attempt to access elements outside the valid range of the array. Without these checks, the program might encounter an **out-of-bounds error**, which could lead to undefined behavior, including crashes or incorrect results.

---

Here is a more elegant solution to the problem. Instead of using two separate loops to find the leftmost and rightmost indices, we can use helper functions to streamline the process. This makes the code cleaner and easier to read.

### Improved Solution

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        return {findBound(nums, target, true), findBound(nums, target, false)};
    }

private:
    int findBound(vector<int>& nums, int target, bool isLeft) {
        int left = 0;
        int right = nums.size() - 1;
        int result = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                result = mid; // Record the current index as a potential result
                if (isLeft) {
                    right = mid - 1; // Narrow down to the left side
                } else {
                    left = mid + 1; // Narrow down to the right side
                }
            } else if (nums[mid] < target) {
                left = mid + 1; // Target is in the right half
            } else {
                right = mid - 1; // Target is in the left half
            }
        }

        return result;
    }
};
```

### Key Improvements
1. **Helper Function for Bounds**:
   - The `findBound` function is used to locate either the leftmost or rightmost index of the target, controlled by the `isLeft` flag. This avoids code duplication and simplifies the logic.

2. **Elegant Binary Search**:
   - By modifying `left` and `right` based on the `isLeft` flag, we efficiently reuse the same binary search logic for both boundaries.

3. **Return Compactly**:
   - The `searchRange` function returns a vector of the leftmost and rightmost indices directly using the results from `findBound`.

### Explanation of the Code
1. **Left Bound Search**:
   - When `isLeft` is `true`, the code searches for the leftmost index of the target by narrowing down the search space to the left (`right = mid - 1`).

2. **Right Bound Search**:
   - When `isLeft` is `false`, the code searches for the rightmost index of the target by narrowing down the search space to the right (`left = mid + 1`).

3. **Time Complexity**:
   - Each binary search operation runs in \( O(\log n) \), and there are two searches, resulting in a total complexity of \( O(\log n) \).

4. **Space Complexity**:
   - The solution uses \( O(1) \) additional space.

### Example Input/Output
#### Input:
```cpp
nums = {5, 7, 7, 8, 8, 10};
target = 8;
```

#### Output:
```cpp
{3, 4}
```

#### Explanation:
- The left bound of `8` is at index `3`.
- The right bound of `8` is at index `4`.

This solution is more concise and modular, making it easier to understand and maintain.