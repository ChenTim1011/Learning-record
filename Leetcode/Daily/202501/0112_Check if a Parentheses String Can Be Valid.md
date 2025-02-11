[Check if a Parentheses String Can Be Valid](https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/description/)

Check from left to right:

Traverse the string, treating every ( and unlocked position (0) as potential open brackets.
If the number of close brackets ) exceeds the available open brackets, the string cannot be valid.
Check from right to left:

Traverse the string in reverse, treating every ) and unlocked position (0) as potential close brackets.
If the number of open brackets ( exceeds the available close brackets, the string cannot be valid.
If both passes succeed, it means the flexible positions can accommodate the required parentheses to balance the string.

Complexity
Time complexity:
O(n) - Each pass (left-to-right and right-to-left) processes the string in linear time.

Space complexity:
O(1) - No additional space is used other than a few integer counters.

```c++
class Solution {
public:
    bool canBeValid(string s, string locked) {
        int n = s.length();
        if (n % 2 != 0) {
            return false; // Odd length cannot form valid parentheses
        }

        // Left-to-right pass: Ensure there are enough open brackets
        int openCount = 0;
        for (int i = 0; i < n; i++) {
            if (s[i] == '(' || locked[i] == '0') {
                openCount++;
            } else { // s[i] == ')' and locked[i] == '1'
                openCount--;
            }
            if (openCount < 0) {
                return false; // Too many ')' encountered
            }
        }

        // Right-to-left pass: Ensure there are enough close brackets
        int closeCount = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (s[i] == ')' || locked[i] == '0') {
                closeCount++;
            } else { // s[i] == '(' and locked[i] == '1'
                closeCount--;
            }
            if (closeCount < 0) {
                return false; // Too many '(' encountered
            }
        }

        return true;
    }
};
```