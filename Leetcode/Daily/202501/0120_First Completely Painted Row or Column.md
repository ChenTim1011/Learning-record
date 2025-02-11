[First Completely Painted Row or Column](https://leetcode.com/problems/first-completely-painted-row-or-column/description/)

### **Problem Key Points**
You need to find the smallest index `i` in `arr` such that either a row or a column in the matrix `mat` is completely painted.

---

### **Solution Approach**

#### 1️⃣ **Create a Mapping for Quick Lookup**
To quickly find the coordinates `(row, col)` of `arr[i]` in the matrix `mat`, a hash map (`positionMap`) is created:
- Traverse the entire matrix `mat` and store the position of each number `(row, col)` in the map.
- This allows O(1) lookup for the position of any number in `arr`.

---

#### 2️⃣ **Track Unpainted Cells in Rows and Columns**
To know when a row or column is fully painted:
- Initialize an array `rowCount` of size `m` (number of rows), where each entry starts with `n` (number of columns). This represents the number of unpainted cells in each row.
- Similarly, initialize an array `colCount` of size `n` (number of columns), where each entry starts with `m` (number of rows). This represents the number of unpainted cells in each column.

---

#### 3️⃣ **Simulate the Painting Process**
Traverse through each number in `arr`:
1. Use the hash map to find the position `(row, col)` of `arr[i]` in `mat`.
2. Decrement the unpainted counts for the corresponding row and column:
   - `rowCount[row]--`
   - `colCount[col]--`
3. Check if any row or column is fully painted (unpainted count becomes zero):
   - If yes, return the current index `i`.
4. If no row or column is fully painted after processing all numbers, return `-1` (although this should not happen due to the problem constraints).

---

### **Code Explanation with Details**

Here’s the C++ implementation:

```cpp
class Solution {
public:
    int firstCompleteIndex(vector<int>& arr, vector<vector<int>>& mat) {
        // Step 1: Initialize matrix dimensions
        int rows = mat.size();    // Number of rows (m)
        int cols = mat[0].size(); // Number of columns (n)

        // Step 2: Create a map from numbers to their positions
        unordered_map<int, pair<int, int>> positionMap;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                positionMap[mat[r][c]] = {r, c};
            }
        }

        // Step 3: Initialize row and column unpainted counts
        vector<int> rowCount(rows, cols); // Each row starts with cols unpainted cells
        vector<int> colCount(cols, rows); // Each column starts with rows unpainted cells

        // Step 4: Traverse arr and simulate the painting process
        for (int idx = 0; idx < arr.size(); ++idx) {
            int val = arr[idx];                  // Get the current value
            auto [row, col] = positionMap[val]; // Find its position in the matrix

            // Decrease the count of unpainted cells for the corresponding row and column
            if (--rowCount[row] == 0 || --colCount[col] == 0) {
                return idx; // If any row or column is fully painted, return the index
            }
        }

        // Step 5: If no row or column is fully painted (should not occur)
        return -1;
    }
};
```

---

### **Why Initialize `rowCount` and `colCount` like This?**

```cpp
vector<int> rowCount(rows, cols);
vector<int> colCount(cols, rows);
```

- `rowCount` is an array of size `rows`, and each element is initialized to `cols` because:
  - Each row initially has `cols` unpainted cells.
- `colCount` is an array of size `cols`, and each element is initialized to `rows` because:
  - Each column initially has `rows` unpainted cells.

For example:
- If `mat` is a 2x3 matrix:
  ```
  mat = [[1, 2, 3],
         [4, 5, 6]]
  ```
  - `rowCount` will be `[3, 3]` because each row starts with 3 unpainted cells.
  - `colCount` will be `[2, 2, 2]` because each column starts with 2 unpainted cells.

---

### **Explanation of the Condition:**

```cpp
if (--rowCount[row] == 0 || --colCount[col] == 0)
```

This condition checks if the current row or column is fully painted:
1. `--rowCount[row]` decreases the unpainted count for the current row by 1.
   - If the count becomes `0`, it means the row is fully painted.
2. `--colCount[col]` decreases the unpainted count for the current column by 1.
   - If the count becomes `0`, it means the column is fully painted.
3. The `||` operator ensures the function returns as soon as either the row or column is completely painted.

---

### **Example Walkthrough**

Let’s take the example:

```cpp
arr = [1, 3, 4, 2];
mat = [[1, 4],
       [2, 3]];
```

#### **Step-by-Step Simulation**

1. **Initialization:**
   - `rowCount = [2, 2]` (2 rows, each with 2 unpainted cells).
   - `colCount = [2, 2]` (2 columns, each with 2 unpainted cells).
   - `positionMap = {1: (0, 0), 4: (0, 1), 2: (1, 0), 3: (1, 1)}`.

2. **Processing `arr[0] = 1`:**
   - Position of `1`: `(0, 0)`.
   - Update counts:
     - `rowCount[0]-- → 1`.
     - `colCount[0]-- → 1`.
   - No row or column is fully painted yet.

3. **Processing `arr[1] = 3`:**
   - Position of `3`: `(1, 1)`.
   - Update counts:
     - `rowCount[1]-- → 1`.
     - `colCount[1]-- → 1`.
   - No row or column is fully painted yet.

4. **Processing `arr[2] = 4`:**
   - Position of `4`: `(0, 1)`.
   - Update counts:
     - `rowCount[0]-- → 0` (Row 0 is fully painted!).
     - `colCount[1]-- → 0` (Column 1 is also fully painted, but row is already detected).
   - Return `idx = 2`.

---

### **Summary**
- The condition `if (--rowCount[row] == 0 || --colCount[col] == 0)` ensures the function returns as soon as a row or column is fully painted.
- Initializing `rowCount` and `colCount` with the number of unpainted cells simplifies tracking and makes the code efficient.