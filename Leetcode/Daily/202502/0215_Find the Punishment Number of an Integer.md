[Find the Punishment Number of an Integer](https://leetcode.com/problems/find-the-punishment-number-of-an-integer/description/)

### Problem Overview

The task is to find the **punishment number** of a given integer \( n \). The punishment number of \( n \) is defined as the sum of the squares of all integers \( i \) such that:

1. \( 1 \leq i \leq n \)
2. The decimal representation of \( i^2 \) can be partitioned into contiguous substrings such that the sum of the integer values of these substrings equals \( i \).

### Example Walkthrough

For example, let's calculate the punishment number for \( n = 10 \).

- \( i = 1 \), \( i^2 = 1 \): The only partition is "1", and \( 1 = 1 \), so it qualifies.
- \( i = 9 \), \( i^2 = 81 \): The partition "8" and "1" gives \( 8 + 1 = 9 \), so it qualifies.
- \( i = 10 \), \( i^2 = 100 \): The partition "10" and "0" gives \( 10 + 0 = 10 \), so it qualifies.

Thus, the punishment number of 10 is \( 1 + 81 + 100 = 182 \).

### Key Insights

- We need to check if the square of each integer \( i \) can be partitioned into substrings whose sum equals \( i \).
- This check involves recursively trying different ways to split the number and summing the parts.

### Solution Approach

To solve this problem, we need to:

1. **Check if a number's square can be partitioned**: This can be done using a recursive function `partition(x, target)`, where we attempt to break the number \( x \) into substrings and check if their sum equals `target` (which is the number \( i \)).

2. **Iterate through all integers from 1 to \( n \)** and calculate their squares. For each square, check if it can be partitioned into valid substrings whose sum equals the integer itself.

3. **Optimize the process** by only considering integers where \( i \mod 9 = 0 \) or \( i \mod 9 = 1 \). This comes from a mathematical observation that only numbers with these properties can potentially form valid partitions for their squares.

4. **Sum the squares of valid integers** and return the result.

### The Code Explanation

The provided solution involves two main parts: the `partition` function and the `punishmentNumber` function.

#### `partition(x, target)` Function

This function checks whether the number \( x \) can be partitioned into contiguous substrings that sum up to the target value \( target \). 

```cpp
static inline bool partition(int x, int target) {
    if (x == target) return true;  // Base case: if x equals target, it's valid
    if (x == 0) return target == 0;  // If x is 0, only a target of 0 is valid
    
    const int m0 = min(x, 1000);  // Limit the range to prevent unnecessary large values
    for (int m = 10; m <= m0; m *= 10) {  // Try to break x into parts by dividing by powers of 10
        if (partition(x / m, target - x % m))  // Recursively check the remaining part
            return true;
    }
    return false;  // If no valid partition is found, return false
}
```

- **Key Idea**: The function works by recursively dividing \( x \) and trying different ways of partitioning it into substrings. It checks whether the sum of the substrings equals the target value.
- **Base Case**: If \( x = target \), return `true` since we've found a valid partition. If \( x \) is 0, we check if the target is also 0.
- **Recursive Case**: For each division of \( x \), it recursively checks if the remaining part of the number can meet the target sum.

#### `punishmentNumber(n)` Function

This function calculates the punishment number for \( n \). It iterates through integers from 1 to \( n \) and checks each one to see if its square can be partitioned in a valid way.

```cpp
static int punishmentNumber(int n) {
    int sum = 0;  // Initialize the sum of valid squares
    for (int i = 1; i <= n; i++) {  // Iterate over each integer from 1 to n
        if (i % 9 != 0 && i % 9 != 1) continue;  // Only consider numbers where i % 9 is 0 or 1
        
        const int x = i * i;  // Compute the square of the number
        sum += (partition(x, i)) ? x : 0;  // If the partition is valid, add the square to the sum
    }
    return sum;  // Return the final sum
}
```

- **Main Idea**: We iterate over all numbers from 1 to \( n \), calculate their squares, and check if the square can be partitioned into valid substrings whose sum equals the number.
- **Optimization**: The check `i % 9 == 0 || i % 9 == 1` is used to reduce unnecessary checks since only numbers with these properties have a potential to meet the partitioning condition.

### Time Complexity

The time complexity of the `partition` function is dependent on how many ways we can partition the number \( x \). The function performs checks recursively, so its complexity is proportional to \( O(\log(x)) \). Since we do this for each number from 1 to \( n \), the overall time complexity is \( O(n \log n) \).

### Conclusion

- **Punishment Number** is the sum of the squares of numbers \( i \) where the square of \( i \) can be partitioned into substrings that sum to \( i \).
- The solution involves recursively checking if a number's square can be split into parts whose sum equals the number itself.
- Optimization techniques (e.g., only checking numbers \( i \mod 9 = 0 \) or \( 1 \)) help reduce unnecessary computations.

This approach is efficient for the given problem constraints and correctly calculates the punishment number for a given \( n \).