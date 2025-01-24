[Rotting Oranges](https://leetcode.com/problems/rotting-oranges/description/)

### Problem Explanation:
You are given a grid where:
- `0` represents an empty cell.
- `1` represents a fresh orange.
- `2` represents a rotten orange.

Every minute, any fresh orange adjacent to a rotten orange (in 4 directions: up, down, left, right) becomes rotten. You need to determine the **minimum time** required for all fresh oranges to become rotten. If it's impossible to rot all oranges, return `-1`.

---

### Intuition:
This problem can be solved using **Breadth-First Search (BFS)**:
- Treat all the initially rotten oranges as the starting points (sources) for the BFS.
- During each step of the BFS, propagate the "rotting effect" to adjacent fresh oranges.
- Keep track of the time elapsed and count the remaining fresh oranges.
- If there are fresh oranges left after the BFS completes, return `-1`.

---

### Approach:

1. **Initialization**:
   - Create a queue (`qe`) to store the positions of all initially rotten oranges.
   - Count the number of fresh oranges (`fresh`).
   - If there are no fresh oranges initially, return `0`.

2. **BFS Traversal**:
   - Use direction vectors to explore neighbors in 4 directions (up, down, left, right).
   - For each rotten orange, check its neighbors. If a neighbor is a fresh orange, rot it, decrement the `fresh` counter, and add it to the queue.

3. **Time Tracking**:
   - Increment the `minutes` counter after processing all oranges in the current layer of the BFS.

4. **Final Check**:
   - If there are no fresh oranges left, return `minutes - 1` (since the last increment is unnecessary after BFS ends).
   - Otherwise, return `-1` if some fresh oranges cannot be reached.

---

### Annotated Code:

```cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        // Handle edge case where the grid is empty
        if (grid.empty()) {
            return 0;
        }
        
        int minutes = 0;       // Tracks the elapsed time
        int fresh = 0;         // Count of fresh oranges
        int m = grid.size();   // Number of rows
        int n = grid[0].size();// Number of columns
        
        // Queue to perform BFS, stores the position of rotten oranges
        queue<pair<int, int>> qe;
        
        // Step 1: Initialize the queue with all rotten oranges and count fresh oranges
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2) {
                    qe.push({i, j}); // Add rotten orange to the queue
                } else if (grid[i][j] == 1) {
                    fresh++; // Count fresh oranges
                }
            }
        }
        
        // If there are no fresh oranges, return 0 (no time needed)
        if (fresh == 0) {
            return 0;
        }
        
        // Direction vectors for exploring neighbors (right, left, up, down)
        vector<pair<int, int>> dir = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        
        // Step 2: Perform BFS
        while (!qe.empty()) {
            int size = qe.size(); // Number of rotten oranges to process at the current level
            
            // Process all oranges at the current level
            while (size--) {
                auto [x, y] = qe.front(); // Get the current rotten orange's position
                qe.pop();
                
                // Explore all 4-directional neighbors
                for (auto& [dx, dy] : dir) {
                    int newX = x + dx; // New row index
                    int newY = y + dy; // New column index
                    
                    // Check if the neighbor is within bounds and is a fresh orange
                    if (newX >= 0 && newX < m && newY >= 0 && newY < n && grid[newX][newY] == 1) {
                        grid[newX][newY] = 2; // Rot the fresh orange
                        qe.push({newX, newY}); // Add it to the queue for the next level
                        fresh--; // Decrease the fresh orange count
                    }
                }
            }
            
            // Increment time after processing the current layer
            minutes++;
        }
        
        // Step 3: Check if there are any fresh oranges left
        return (fresh == 0) ? minutes - 1 : -1;
    }
};
```

---

### Explanation of Key Steps:

1. **Initialization**:
   - Rotten oranges are added to the queue so that BFS starts from them.
   - Fresh oranges are counted so we can check at the end if all were processed.

2. **BFS**:
   - BFS ensures that oranges are processed layer by layer (minute by minute).
   - The queue size at each step represents the number of rotten oranges at that minute.

3. **Time Increment**:
   - Each level of BFS corresponds to 1 minute. We increment `minutes` after processing all rotten oranges at the current level.

4. **Final Check**:
   - If there are no fresh oranges left, return the total time taken.
   - If there are still fresh oranges, return `-1` as it's impossible to rot all oranges.

---

### Complexity:

1. **Time Complexity**:
   - Each cell is processed at most once during BFS.
   - Total operations = \(O(m \times n)\), where \(m\) is the number of rows and \(n\) is the number of columns.

2. **Space Complexity**:
   - The queue can hold at most \(O(m \times n)\) elements in the worst case.
   - Direction vectors require constant space \(O(1)\).

---

### Examples:

#### Example 1:
**Input**: 
```plaintext
grid = [[2,1,1],
        [1,1,0],
        [0,1,1]]
```
**Output**: `4`

#### Example 2:
**Input**: 
```plaintext
grid = [[2,1,1],
        [0,1,1],
        [1,0,1]]
```
**Output**: `-1`

#### Example 3:
**Input**: 
```plaintext
grid = [[0,2]]
```
**Output**: `0`

---
