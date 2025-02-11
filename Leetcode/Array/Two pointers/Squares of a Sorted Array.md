[Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/description/)

## Intuition method

```c++
```c++
// Method 1: Sorting after squaring
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        // Input: nums = [-4, -1, 0, 3, 10]
        // Output: [0, 1, 9, 16, 100]

        // Step 1: Take the absolute value of each element
        for(int i = 0; i < nums.size(); i++) {
            nums[i] = abs(nums[i]);
        }

        // Step 2: Sort the array
        sort(nums.begin(), nums.end());

        // Step 3: Square each element
        for(int i = 0; i < nums.size(); i++) {
            nums[i] = nums[i] * nums[i];
        }

        // Return the sorted squares
        return nums;
    }
};

/*
Explanation:
1. The first loop converts all elements to their absolute values.
2. The array is then sorted.
3. Each element is squared.
4. The result is a sorted array of squares.
5. Time complexity: O(n log n) due to sorting.
6. Space complexity: O(1) if we ignore the space used by the input and output vectors.
*/
```

## Two pointer method

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        // Input: nums = [-4, -1, 0, 3, 10]
        // Output: [0, 1, 9, 16, 100]

        // Initialize the output vector with the same size as nums, filled with 0s
        vector<int> output(nums.size(), 0);

        // Two pointers starting from the beginning and end of the array
        int left = 0;
        int right = nums.size() - 1;
        int index = nums.size() - 1; // Index to place the next largest square

        // While the left pointer is less than or equal to the right pointer
        while (left <= right) {
            // Compare the absolute values of the elements at the pointers
            if (abs(nums[right]) >= abs(nums[left])) {
                // If the right element is larger or equal, square it and place it in the output
                output[index--] = nums[right] * nums[right];
                right--; // Move the right pointer to the left
            } else {
                // If the left element is larger, square it and place it in the output
                output[index--] = nums[left] * nums[left];
                left++; // Move the left pointer to the right
            }
        }

        // Return the sorted squares
        return output;
    }
};

/*
Explanation:
1. Initialize two pointers, one at the start (left) and one at the end (right) of the array.
2. Compare the absolute values of the elements at the pointers.
3. Place the larger square at the current index in the output array and move the corresponding pointer.
4. Repeat until all elements are processed.
5. This method avoids the need to sort the array, resulting in a time complexity of O(n).
6. Space complexity: O(n) for the output array.
*/
```