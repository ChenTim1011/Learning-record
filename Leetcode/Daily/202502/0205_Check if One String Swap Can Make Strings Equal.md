[Check if One String Swap Can Make Strings Equal](https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/description/)

### Code:
```cpp
class Solution {
public:
    bool areAlmostEqual(string s1, string s2) {
        if (s1 == s2) return true; // If the strings are already equal, return true

        vector<int> diff; // To store indices where characters differ
        for (int i = 0; i < s1.size(); i++) {
            if (s1[i] != s2[i]) diff.push_back(i); // Find differing indices
        }

        // Check if the difference can be resolved by a single swap
        return diff.size() == 2 && 
               s1[diff[0]] == s2[diff[1]] && 
               s1[diff[1]] == s2[diff[0]];
    }
};
```


### Explanation:

#### 1. **Check if Strings are Already Equal**:
   - The first `if` statement checks whether `s1` and `s2` are already equal.
   - If they are, there's no need to perform a swap, so it directly returns `true`.

#### 2. **Find Differing Indices**:
   - The `for` loop compares characters of `s1` and `s2` at each index.
   - If a mismatch is found (i.e., `s1[i] != s2[i]`), the index `i` is added to the `diff` vector.

#### 3. **Check Conditions for One Swap**:
   - The `diff` vector will contain the indices where the two strings differ.
   - For the strings to be made equal with one swap:
     1. There must **exactly** be **two mismatched indices** (size of `diff` must be 2).
     2. The characters at those mismatched indices must be "swappable," i.e.:
        - `s1[diff[0]] == s2[diff[1]]`
        - `s1[diff[1]] == s2[diff[0]]`
   - If these conditions are met, the function returns `true`.
   - Otherwise, return `false`.

---

### Key Improvements Over the Original Code:

1. **Avoids Unnecessary Operations**:
   - Your original code involved counting character occurrences and modifying `s2`. These operations are unnecessary since we only need to find mismatched indices and check swapability.

2. **Efficiently Handles Edge Cases**:
   - This solution directly handles cases where:
     - The strings are already equal.
     - The number of mismatched characters is not exactly two.
   - This eliminates redundant checks and simplifies logic.

3. **Readable and Concise**:
   - The logic is compact and easy to understand. The conditions for a valid swap are explicitly defined.

---

### Example Walkthrough:

#### **Example 1**:
Input: `s1 = "bank", s2 = "kanb"`

- Mismatched indices: `diff = [0, 3]`
- Characters at these indices:
  - `s1[0] = 'b'`, `s2[3] = 'b'`
  - `s1[3] = 'k'`, `s2[0] = 'k'`
- These characters can be swapped to make the strings equal, so the output is `true`.

#### **Example 2**:
Input: `s1 = "attack", s2 = "defend"`

- Mismatched indices: `diff` contains more than 2 indices.
- Since more than one swap would be needed, the output is `false`.

#### **Example 3**:
Input: `s1 = "kelb", s2 = "kelb"`

- The strings are already equal (`s1 == s2`), so the output is `true`.

---

### Complexity Analysis:

- **Time Complexity**: \(O(n)\), where \(n\) is the length of the strings.
  - We only iterate through the strings once to find mismatched indices.
- **Space Complexity**: \(O(1)\) (excluding the `diff` vector size), as we only store up to two indices in `diff`.

