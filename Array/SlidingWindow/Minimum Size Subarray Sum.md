[Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/)

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        // Initialize variables
        int sum = 0; // Current sum of the subarray
        int head = 0; // End pointer of the subarray
        int tail = 0; // Start pointer of the subarray
        int result = INT_MAX; // Result to store the minimum length of subarray

        // Iterate over the array with the head pointer
        for (head = 0; head < nums.size(); head++) {
            sum += nums[head]; // Add the current element to the sum

            // While the current sum is greater than or equal to the target
            while (sum >= target) {
                int sublength = head - tail + 1; // Calculate the length of the current subarray
                result = min(result, sublength); // Update the result with the minimum length
                sum -= nums[tail]; // Remove the element at the tail from the sum
                tail++; // Move the tail pointer to the right
            }
        }

        // If result is still INT_MAX, it means no valid subarray was found
        if (result == INT_MAX) return 0;
        return result; // Return the minimum length of the subarray
    }
};
```