[Binary Search](https://leetcode.com/problems/binary-search/description/)

### Approach

The premise of this problem is that the array is a **sorted array**, and the problem explicitly states that there are **no duplicate elements**. This is critical because, with duplicates, the index returned by binary search might not be unique. These conditions are prerequisites for using the binary search algorithm. When you see that a problem satisfies these conditions, think about whether binary search is applicable.

Binary search often involves a lot of edge cases. While the logic is simple, it can be tricky to write correctly. For instance:
- Should the loop condition be `while(left < right)` or `while(left <= right)`?
- Should `right` be updated to `middle` or `middle - 1`?

The confusion in implementing binary search typically arises from not having a clear understanding of the **definition of the interval**. The interval definition is the **invariant** of the binary search. During the binary search process, the invariant must be maintained, meaning that each boundary update must adhere to the interval's definition. This is the **loop invariant rule**.

When writing binary search, the interval is typically defined in two ways:
1. **Left-closed, right-closed**: `[left, right]`
2. **Left-closed, right-open**: `[left, right)`

Below, I will explain two different implementations of binary search based on these interval definitions.

---

### Binary Search: First Implementation

In this implementation, we define the target to be within a **left-closed, right-closed interval**: `[left, right]` (this is very important).

This interval definition determines how the binary search code should be written. Given the definition of the target in `[left, right]`, the following rules apply:
1. Use `while (left <= right)` because `left == right` is meaningful in this interval, so `<=` is used.
2. If `nums[middle] > target`, update `right = middle - 1` because the current `nums[middle]` cannot be the target. The end index of the search interval for the left subarray is therefore `middle - 1`.

For example, to search for the element `2` in the array `[1, 2, 3, 4, 7, 9, 10]`, refer to the illustration:

**Binary Search Example: First Implementation**

Code (detailed comments included):

```cpp
// Version 1
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1; // Define the target in the interval [left, right]
        while (left <= right) { // When left == right, the interval [left, right] is still valid, so use <=
            int middle = left + ((right - left) / 2); // Prevent overflow, equivalent to (left + right) / 2
            if (nums[middle] > target) {
                right = middle - 1; // Target is in the left interval: [left, middle - 1]
            } else if (nums[middle] < target) {
                left = middle + 1; // Target is in the right interval: [middle + 1, right]
            } else { // nums[middle] == target
                return middle; // Target found, return the index
            }
        }
        // Target not found
        return -1;
    }
};
```

**Time Complexity**: \( O(\log n) \)  
**Space Complexity**: \( O(1) \)

---

### Binary Search: Second Implementation

In this implementation, the target is defined to be within a **left-closed, right-open interval**: `[left, right)`. The boundary handling is quite different from the first implementation.

The rules are as follows:
1. Use `while (left < right)` because `left == right` is meaningless in the interval `[left, right)`. Hence, `<` is used.
2. If `nums[middle] > target`, update `right = middle`. Since `nums[middle]` is not equal to the target, search the left interval. As the interval is left-closed, right-open, the new right boundary becomes `middle`.

For example, to search for the element `2` in the array `[1, 2, 3, 4, 7, 9, 10]`, refer to the illustration:  
**Binary Search Example: Second Implementation**  

Code (detailed comments included):

```cpp
// Version 2
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size(); // Define the target in the interval [left, right)
        while (left < right) { // When left == right, the interval [left, right) is invalid, so use <
            int middle = left + ((right - left) >> 1);
            if (nums[middle] > target) {
                right = middle; // Target is in the left interval: [left, middle)
            } else if (nums[middle] < target) {
                left = middle + 1; // Target is in the right interval: [middle + 1, right)
            } else { // nums[middle] == target
                return middle; // Target found, return the index
            }
        }
        // Target not found
        return -1;
    }
};
```

**Time Complexity**: \( O(\log n) \)  
**Space Complexity**: \( O(1) \)

