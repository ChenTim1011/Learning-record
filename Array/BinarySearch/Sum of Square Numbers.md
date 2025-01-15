[Sum of Square Numbers](https://leetcode.com/problems/sum-of-square-numbers/description/)


```c++
class Solution {
public:
    bool judgeSquareSum(int c) {
        // Handle negative input
        if (c < 0) {
            throw std::invalid_argument("Input must be non-negative");
        }

        // Handle edge case of c = 0 (0^2 + 0^2 = 0)
        if (c == 0) return true;

        // Use long to prevent integer overflow
        // sqrt returns double, so we need to cast it to long
        long right = static_cast<long>(sqrt(c));
        
        // Optimization: No need to check numbers larger than sqrt(c)
        // since their squares would exceed c
        long left = 0;

        // Optimization: Cache the squares to avoid repeated multiplication
        long left_square = 0;
        long right_square = right * right;

        while (left <= right) {
            // Calculate sum using cached squares
            long curr_sum = left_square + right_square;

            if (curr_sum == c) {
                return true;
            } else if (curr_sum < c) {
                left++;
                // Update left square cache
                left_square = left * left;
                
                // Check for overflow
                if (left_square < 0) {
                    throw std::overflow_error("Integer overflow occurred");
                }
            } else {
                right--;
                // Update right square cache
                right_square = right * right;
            }
        }

        return false;
    }
};
};
```