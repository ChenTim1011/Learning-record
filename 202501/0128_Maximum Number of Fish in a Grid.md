[Maximum Number of Fish in a Grid](https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/description/)

### Problem Explanation
The goal is to find the **maximum number of fish** a fisher can catch in the grid by starting at any water cell and collecting fish from all connected water cells. A water cell is a cell where `grid[r][c] > 0`. The fisher can only move to adjacent cells (up, down, left, or right).

### Key Observations:
1. The grid consists of "land cells" (`grid[r][c] == 0`) and "water cells" (`grid[r][c] > 0`).
2. A connected group of water cells forms a "region" (like islands in a graph).
3. You need to calculate the total number of fish in each connected region and return the maximum among all regions.

We can solve this problem using **graph traversal algorithms**:
1. **Breadth-First Search (BFS)**.
2. **Depth-First Search (DFS)**.

---

### **BFS Solution**
The BFS approach uses a queue to explore all connected water cells from a starting cell.

#### Code with Explanation
```cpp
class Solution {
public:
    int findMaxFish(vector<vector<int>>& grid) {
        int m = grid.size(); // Number of rows
        int n = grid[0].size(); // Number of columns
        vector<vector<bool>> visited(m, vector<bool>(n, false)); // To track visited cells
        vector<pair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}}; // Four directions: right, left, down, up
        int maxFish = 0; // To store the maximum fish count found

        // BFS function to calculate the total fish in a connected region
        auto bfs = [&](int startX, int startY) {
            queue<pair<int, int>> q; // Queue to perform BFS
            q.push({startX, startY}); // Start BFS from this cell
            visited[startX][startY] = true; // Mark the cell as visited
            int fishCount = 0; // Initialize the fish count for this region

            while (!q.empty()) {
                auto [x, y] = q.front();
                q.pop();
                fishCount += grid[x][y]; // Add fish in the current cell

                // Explore all 4 directions
                for (auto [dx, dy] : directions) {
                    int newX = x + dx;
                    int newY = y + dy;

                    // Check if the new cell is within bounds, unvisited, and contains fish
                    if (newX >= 0 && newX < m && newY >= 0 && newY < n && grid[newX][newY] > 0 && !visited[newX][newY]) {
                        visited[newX][newY] = true;
                        q.push({newX, newY}); // Add the new cell to the queue
                    }
                }
            }
            return fishCount; // Return the total fish collected in this region
        };

        // Traverse the grid to find all water cells and calculate the maximum fish
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] > 0 && !visited[i][j]) { // If this is an unvisited water cell
                    maxFish = max(maxFish, bfs(i, j)); // Perform BFS and update the maximum fish
                }
            }
        }
        return maxFish; // Return the maximum fish found
    }
};
```

---

### **DFS Solution**
The DFS approach uses recursion to explore all connected water cells from a starting cell.

#### Code with Explanation
```cpp
class Solution {
public:
    int findMaxFish(vector<vector<int>>& grid) {
        int m = grid.size(); // Number of rows
        int n = grid[0].size(); // Number of columns
        vector<vector<bool>> visited(m, vector<bool>(n, false)); // To track visited cells
        vector<pair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}}; // Four directions: right, left, down, up
        int maxFish = 0; // To store the maximum fish count found

        // DFS function to calculate the total fish in a connected region
        function<int(int, int)> dfs = [&](int x, int y) {
            visited[x][y] = true; // Mark the cell as visited
            int fishCount = grid[x][y]; // Add fish in the current cell

            // Explore all 4 directions
            for (auto [dx, dy] : directions) {
                int newX = x + dx;
                int newY = y + dy;

                // Check if the new cell is within bounds, unvisited, and contains fish
                if (newX >= 0 && newX < m && newY >= 0 && newY < n && grid[newX][newY] > 0 && !visited[newX][newY]) {
                    fishCount += dfs(newX, newY); // Recursively visit the new cell
                }
            }
            return fishCount; // Return the total fish collected in this region
        };

        // Traverse the grid to find all water cells and calculate the maximum fish
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] > 0 && !visited[i][j]) { // If this is an unvisited water cell
                    maxFish = max(maxFish, dfs(i, j)); // Perform DFS and update the maximum fish
                }
            }
        }
        return maxFish; // Return the maximum fish found
    }
};
```

---

### Example Walkthrough
**Input:**  
`grid = [[0,2,1,0], [4,0,0,3], [1,0,0,4], [0,3,2,0]]`

#### BFS Walkthrough:
1. Start at cell `(0,1)` (fish = 2). Visit `(0,2)` (fish = 1). Total fish = 3.
2. Start at cell `(1,0)` (fish = 4). Visit `(2,0)` (fish = 1). Total fish = 5.
3. Start at cell `(1,3)` (fish = 3). Visit `(2,3)` (fish = 4). Total fish = 7.
4. Start at cell `(3,1)` (fish = 3). Visit `(3,2)` (fish = 2). Total fish = 5.

**Maximum Fish = 7** (from region starting at `(1,3)`).

---

### Complexity Analysis
#### Time Complexity:
- **BFS and DFS**: `O(m * n)` since each cell is visited once.
#### Space Complexity:
- BFS: `O(m * n)` for the queue and visited array.
- DFS: `O(m * n)` for the recursion stack and visited array.

Both solutions are efficient given the constraints (`m, n ≤ 10`). Let me know if you'd like further clarifications!