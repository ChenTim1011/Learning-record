[Find Missing and Repeated Values](https://leetcode.com/problems/find-missing-and-repeated-values/description/)


## **📌 Problem Statement**
Given an `n x n` grid where:
- All numbers from `1` to `n^2` should appear **exactly once**.
- However, **one number appears twice** (`a`), and **one number is missing** (`b`).

We need to return `[a, b]`, where:
- `a` = The **repeated** number.
- `b` = The **missing** number.

---

## **🔹 Examples**
### **Example 1**
```
Input: grid = [[1,3],[2,2]]
Output: [2,4]
Explanation:
- The numbers **1, 2, 2, 3** appear in the grid.
- **2 is repeated** and **4 is missing**.
- So, return **[2,4]**.
```

### **Example 2**
```
Input: grid = [[9,1,7],[8,9,2],[3,4,6]]
Output: [9,5]
Explanation:
- The numbers **1, 2, 3, 4, 6, 7, 8, 9, 9** appear in the grid.
- **9 is repeated** and **5 is missing**.
- So, return **[9,5]**.
```

---

## **🔹 Approach 1: Use a Frequency Array (O(n²) Time, O(n²) Space)**
### **✅ Steps**
1. **Create a frequency array** `freq` of size `n² + 1`, initialized to `0`.
2. **Iterate through the grid** and update `freq[num]`.
3. **Find the duplicate and missing values**:
   - The number with `freq[i] == 2` is the **duplicate** (`a`).
   - The number with `freq[i] == 0` is the **missing** (`b`).

### **💡 Code**
```cpp
class Solution {
public:
    vector<int> findMissingAndRepeatedValues(vector<vector<int>>& grid) {
        int n = grid.size();
        int size = n * n;
        vector<int> freq(size + 1, 0);  // Frequency array for numbers [1, n^2]
        int a = -1, b = -1;

        // Count frequencies
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                freq[grid[i][j]]++;
            }
        }

        // Find the repeated and missing numbers
        for (int i = 1; i <= size; i++) {
            if (freq[i] == 2) a = i;  // Duplicate
            if (freq[i] == 0) b = i;  // Missing
        }

        return {a, b};
    }
};
```

### **⏳ Complexity Analysis**
| Step | Time Complexity | Space Complexity |
|------|---------------|----------------|
| Count frequencies | `O(n²)` | `O(n²)` (for `freq` array) |
| Find missing & duplicate | `O(n²)` | `O(1)` |
| **Overall** | `O(n²)` | `O(n²)` |

✅ **Works well for small constraints (`n ≤ 50`) but uses extra space.**

