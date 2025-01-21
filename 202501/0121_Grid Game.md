[Grid game](https://leetcode.com/problems/grid-game/description/)


We are tasked with solving the **Grid Game** problem. The grid is a 2x`n` matrix where each cell `grid[r][c]` contains some points. Two robots traverse the grid as follows:
1. **First Robot's Path:**
   - Starts at `(0, 0)` (top-left corner).
   - Ends at `(1, n-1)` (bottom-right corner).
   - Can only move **right** or **down**.
   - Collects points from the cells it visits, and sets those cells to `0`.
2. **Second Robot's Path:**
   - Starts at `(0, 0)` (top-left corner).
   - Ends at `(1, n-1)` (bottom-right corner).
   - Can only move **right** or **down**.
   - Collects points from the cells it visits after the first robot has already modified the grid.
3. The **Goal:**
   - The **first robot** plays optimally to minimize the points collected by the **second robot**.
   - The **second robot** plays optimally to maximize the points it collects.

The output should be the maximum points the **second robot** can collect after both robots have played optimally.

---

### Solution Approach: Step-by-Step Explanation

#### Step 1: Understand Key Observations
1. **First Robot's Path Determines the Outcome:**
   - The first robot's path creates two regions:
     - **Top remaining points:** Above the path of the first robot.
     - **Bottom accumulated points:** Below the path of the first robot.
   - The second robot will choose the region (top or bottom) that gives it the most points.
2. **First Robot's Strategy:**
   - Minimize the maximum of the two regions.
   - This ensures that the second robot collects the least possible points.

#### Step 2: Compute Prefix Sums
1. Calculate prefix sums for the **top row** and **bottom row**:
   - `row1sum`: Total points remaining in the top row.
   - `row2sum`: Total points accumulated in the bottom row.

#### Step 3: Simulate First Robot's Moves
1. As the first robot moves column by column:
   - Update `row1sum` to exclude the current column's points.
   - Update `row2sum` to include the current column's points.
2. At each column:
   - Calculate the **maximum points** the second robot can collect:
     - From the top: `row1sum` (remaining points in the top row after this column).
     - From the bottom: `row2sum` (accumulated points in the bottom row up to this column).
   - Keep track of the **minimum of these maximums** (min-max strategy).

#### Step 4: Return the Result
1. The result is the smallest value of the maximum points the second robot can collect.

---

### Implementation: Code Explanation
Here's the implementation of the above approach:

```cpp
class Solution {
public:
    long long gridGame(vector<vector<int>>& grid) {
        int n = grid[0].size();  // Length of the grid.
        
        // Step 1: Calculate prefix sums for both rows.
        long long row1sum = accumulate(grid[0].begin(), grid[0].end(), 0LL);  // Total points in the top row.
        long long row2sum = 0;  // Total points in the bottom row starts at 0.
        
        long long result = LLONG_MAX;  // Initialize the minimum result to a large value.
        
        // Step 2: Simulate the first robot's path column by column.
        for (int i = 0; i < n; i++) {
            // Update row1sum to exclude points the first robot has passed.
            row1sum -= grid[0][i];
            
            // Calculate the maximum points the second robot can collect.
            long long secondRobotPoints = max(row1sum, row2sum);
            
            // Update the minimum of these maximums.
            result = min(result, secondRobotPoints);
            
            // Update row2sum to include points the first robot has passed.
            row2sum += grid[1][i];
        }
        
        return result;  // Return the minimum maximum points.
    }
};
```

---

### Example Walkthrough

#### Input: `grid = [[2,5,4],[1,5,1]]`
1. **Initial Calculations:**
   - `row1sum = 2 + 5 + 4 = 11`
   - `row2sum = 0`
   - `result = LLONG_MAX`

2. **Column-by-Column Simulation:**
   - **Column 0:**
     - Update `row1sum = 11 - 2 = 9`
     - `secondRobotPoints = max(9, 0) = 9`
     - Update `result = min(LLONG_MAX, 9) = 9`
     - Update `row2sum = 0 + 1 = 1`
   - **Column 1:**
     - Update `row1sum = 9 - 5 = 4`
     - `secondRobotPoints = max(4, 1) = 4`
     - Update `result = min(9, 4) = 4`
     - Update `row2sum = 1 + 5 = 6`
   - **Column 2:**
     - Update `row1sum = 4 - 4 = 0`
     - `secondRobotPoints = max(0, 6) = 6`
     - Update `result = min(4, 6) = 4`
     - Update `row2sum = 6 + 1 = 7`

3. **Output:**
   - The minimum maximum points the second robot can collect is `4`.

---

### Complexity Analysis

#### Time Complexity:
1. **Prefix Sum Calculation:** `O(n)`
2. **Column Simulation Loop:** `O(n)`
   - Update values and calculate min-max for each column.
3. **Overall:** `O(n)`

#### Space Complexity:
1. No extra space is used aside from a few variables (`O(1)`).

**Final Complexity:**  
- **Time:** `O(n)`
- **Space:** `O(1)`

---

### Key Takeaways
- The first robot uses a **min-max strategy** to minimize the second robot's maximum score.
- Efficient calculation of prefix sums and dynamic updates are key to achieving `O(n)` time complexity.