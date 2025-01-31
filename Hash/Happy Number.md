[Happy Number](https://leetcode.com/problems/happy-number/description/)

### Problem Explanation: Happy Number

A **happy number** is defined by a specific process:
1. Start with any positive integer \( n \).
2. Replace \( n \) with the sum of the squares of its digits.
3. Repeat this process:
   - If \( n \) eventually becomes 1, it is a happy number.
   - If \( n \) falls into a cycle (repeating values that do not include 1), it is not a happy number.

The task is to determine whether a given number \( n \) is a happy number or not.

---

### Examples:

#### Example 1:
Input:  
`n = 19`  
Output:  
`true`  

Explanation:  
1. \( 1^2 + 9^2 = 82 \)  
2. \( 8^2 + 2^2 = 68 \)  
3. \( 6^2 + 8^2 = 100 \)  
4. \( 1^2 + 0^2 + 0^2 = 1 \)  

Since the process reaches 1, \( 19 \) is a happy number.

---

#### Example 2:
Input:  
`n = 2`  
Output:  
`false`  

Explanation:  
1. \( 2^2 = 4 \)  
2. \( 4^2 = 16 \)  
3. \( 1^2 + 6^2 = 37 \)  
4. \( 3^2 + 7^2 = 58 \)  
5. \( 5^2 + 8^2 = 89 \)  
6. \( 8^2 + 9^2 = 145 \)  
7. \( 1^2 + 4^2 + 5^2 = 42 \)  
8. \( 4^2 + 2^2 = 20 \)  
9. \( 2^2 + 0^2 = 4 \)  

At this point, the process enters a loop (4 → 16 → 37 → ...), so \( 2 \) is **not** a happy number.

---

### Constraints:
1. \( 1 \leq n \leq 2^{31} - 1 \) (Any positive integer up to the maximum 32-bit integer.)

---

### Algorithm and Explanation:

1. **Break Down the Digits**:
   - Extract each digit of the number \( n \) and compute the square of the digit. Sum these squares to form the next number.

2. **Track Seen Numbers**:
   - Use a set to store all previously computed numbers.
   - If the current number \( n \) has already been seen, a cycle exists, and the number is not happy.
   - If \( n = 1 \), the number is happy.

3. **Loop Until Result**:
   - Continue this process until \( n = 1 \) (return true) or a cycle is detected (return false).

---

### Annotated Code:

```cpp
class Solution {
public:
    // Helper function to compute the sum of squares of digits of a number
    int getsum(int n) {
        int sum = 0; // Variable to store the sum of squares
        while (n > 0) {
            int digit = n % 10; // Extract the last digit
            sum += digit * digit; // Add the square of the digit to the sum
            n /= 10; // Remove the last digit
        }
        return sum; // Return the computed sum of squares
    }

    // Main function to check if a number is happy
    bool isHappy(int n) {
        unordered_set<int> nums; // Set to store numbers we've seen to detect cycles
        
        while (true) {
            int happynumber = getsum(n); // Compute the sum of squares of digits
            if (happynumber == 1) { // If we reach 1, the number is happy
                return true;
            } else {
                // If we've already seen this number, it's a cycle
                if (nums.find(happynumber) != nums.end()) {
                    return false; // Not a happy number
                } else {
                    nums.insert(happynumber); // Add the number to the set
                }
            }
            n = happynumber; // Update n to the new number
        }
        return false; // Should not reach here (included for completeness)
    }
};
```

---

### Explanation of the Code:

#### **Helper Function: `getsum`**
- Takes a number \( n \) as input.
- Computes the sum of the squares of its digits:
  1. Use \( n \% 10 \) to extract the last digit.
  2. Add the square of the digit to `sum`.
  3. Use \( n /= 10 \) to remove the last digit.
- Returns the computed sum.

#### **Main Function: `isHappy`**
1. Uses a set `nums` to track numbers we've already encountered.
2. Enters a loop:
   - Computes the sum of squares of digits using `getsum`.
   - If the result is 1, the number is happy (`return true`).
   - If the result is already in the set `nums`, a cycle exists (`return false`).
   - Otherwise, adds the result to `nums` and continues.
3. The process continues until either \( n = 1 \) or a cycle is detected.

---

### Example Walkthrough:

#### Input:
`n = 19`

1. Initialize: `nums = {}`.
2. Compute `getsum(19) → 82`. Add 82 to `nums`.
3. Compute `getsum(82) → 68`. Add 68 to `nums`.
4. Compute `getsum(68) → 100`. Add 100 to `nums`.
5. Compute `getsum(100) → 1`. Return `true`.

#### Input:
`n = 2`

1. Initialize: `nums = {}`.
2. Compute `getsum(2) → 4`. Add 4 to `nums`.
3. Compute `getsum(4) → 16`. Add 16 to `nums`.
4. Compute `getsum(16) → 37`. Add 37 to `nums`.
5. Compute `getsum(37) → 58`. Add 58 to `nums`.
6. Compute `getsum(58) → 89`. Add 89 to `nums`.
7. Compute `getsum(89) → 145`. Add 145 to `nums`.
8. Compute `getsum(145) → 42`. Add 42 to `nums`.
9. Compute `getsum(42) → 20`. Add 20 to `nums`.
10. Compute `getsum(20) → 4`. Detected a cycle (4 already in `nums`). Return `false`.

---

### Complexity Analysis:

1. **Time Complexity**:
   - Each number has at most \( \log_{10}(n) \) digits, and we compute the sum of squares for each digit.
   - The set lookup and insert operations are \( O(1) \).
   - The number of iterations is bounded because there are only so many possible sums of squares for numbers (maximum value for a 9-digit number is 729).
   - **Overall**: \( O(\log_{10}(n)) \) per iteration, with a bounded number of iterations.

2. **Space Complexity**:
   - The set `nums` stores unique sums of squares.
   - Maximum space usage depends on the number of unique sums.
   - **Overall**: \( O(U) \), where \( U \) is the number of unique sums of squares.

---

This code ensures correctness and efficiency while handling both small and large inputs effectively.