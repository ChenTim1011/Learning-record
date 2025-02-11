[Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/description/)

```c++
class Solution {
public:
    bool isPerfectSquareBinarySearch(int num) {
        // Handle invalid input
        if (num < 0) {
            throw std::invalid_argument("Input must be non-negative");
        }

        // Handle edge cases
        if (num == 0 || num == 1) return true;

        // Binary search implementation
        // Optimize initial range: right can start at num/2 + 1
        // since sqrt(num) ≤ num/2 for num > 4
        long long left = 1;
        long long right = (num / 2) + 1;

        while (left <= right) {
            long long mid = left + (right - left) / 2;  // Prevent overflow
            long long square = mid * mid;

            // Check for overflow in square calculation
            if (mid > 0 && square / mid != mid) {
                throw std::overflow_error("Integer overflow in square calculation");
            }

            if (square == num) {
                return true;
            }
            if (square < num) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }
};
```