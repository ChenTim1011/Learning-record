[Neighboring Bitwise XOR](https://leetcode.com/problems/neighboring-bitwise-xor/description/)


# Problem Understanding

The problem "Neighboring Bitwise XOR" deals with binary arrays and XOR operations between adjacent elements. Let's break it down:

1. We have a derived array that was created from an original binary array using XOR operations
2. For each element i in derived:
   - If it's not the last element: derived[i] = original[i] ⊕ original[i+1]
   - If it's the last element: derived[n-1] = original[n-1] ⊕ original[0]
3. We need to determine if it's possible for any valid original array to exist that could create the given derived array

# Why XOR Sum Must Be 0

Let's understand why the XOR sum of all elements in derived must be 0 for a valid solution:

1. Consider what happens when we XOR all elements in derived:
```
derived[0] = original[0] ⊕ original[1]
derived[1] = original[1] ⊕ original[2]
derived[2] = original[2] ⊕ original[3]
...
derived[n-1] = original[n-1] ⊕ original[0]
```

2. When we XOR all these equations:
   - Each original[i] appears exactly twice (once with i-1 and once with i+1)
   - When you XOR the same number twice, it cancels out (a ⊕ a = 0)
   - Therefore, all original values must cancel out

Let's take Example 1: derived = [1,1,0]

```cpp
// Example walkthrough:
original = [0,1,0]  // A possible valid solution

// Computing derived:
derived[0] = 0 ⊕ 1 = 1
derived[1] = 1 ⊕ 0 = 1
derived[2] = 0 ⊕ 0 = 0

// XOR sum of derived:
1 ⊕ 1 ⊕ 0 = 0  // Valid because sum is 0
```

# Code Explanation

```cpp
class Solution {
public:
    bool doesValidArrayExist(vector<int>& derived) {
        int xr = 0;
        for (int x : derived) {
            xr ^= x;  // Compute running XOR of all elements
        }
        return xr == 0;  // Check if final XOR is 0
    }
};
```

The solution is remarkably simple:
1. Initialize a variable `xr` to track the running XOR sum
2. Iterate through each element in derived, XORing it with `xr`
3. Return true if the final XOR sum is 0, false otherwise

# Why This Works

- If the XOR sum is 0, we can always construct a valid original array
- Starting with original[0] = 0, we can work forward using the derived values to determine each next value
- If XOR sum is not 0, it's impossible to have a valid original array because the values won't "close the loop" properly

# Example Walkthrough

For derived = [1,1,0]:
```
Initial xr = 0
xr ^= 1  // xr = 1
xr ^= 1  // xr = 0
xr ^= 0  // xr = 0
Return true because xr == 0
```

The solution has O(n) time complexity as it makes a single pass through the array, and O(1) space complexity as it only uses a single variable regardless of input size.