[01 Matrix](https://leetcode.com/problems/01-matrix/description/)

---

### Problem Explanation:
The task is to find the distance of each cell in a binary matrix to the nearest cell containing `0`. The distance between two adjacent cells is `1`.

---

### Intuition:
We can solve this problem using **Breadth-First Search (BFS)**. BFS is particularly suited for this task because it explores all cells layer by layer, ensuring that the shortest path to each cell is calculated.

---

### Approach:

1. **Initialization**:
   - Create a distance matrix (`dis`) of the same size as the input matrix, initialized to `-1`. This helps identify unvisited cells.
   - Use a queue to perform BFS. Add all cells with a `0` to the queue and set their distances in `dis` to `0`.

2. **BFS Traversal**:
   - Process each cell in the queue and explore its neighbors (up, down, left, right).
   - If a neighbor is valid (i.e., within bounds and unvisited), update its distance as the current cell's distance + 1, and add it to the queue.

3. **Direction Control**:
   - Use direction arrays (`dir`) to simplify the logic for exploring neighbors. This eliminates the need for manual if-else conditions for each direction.

4. **Result**:
   - After the BFS is complete, the `dis` matrix contains the shortest distance for each cell to the nearest `0`.

---

### Annotated Code:

```cpp
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int m = mat.size(); // Number of rows
        int n = mat[0].size(); // Number of columns
        
        // Distance matrix initialized to -1 (unvisited cells)
        vector<vector<int>> dis(m, vector<int>(n, -1));
        
        // Queue to perform BFS
        queue<pair<int, int>> qe;

        // Add all cells containing 0 to the queue and set their distance to 0
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) {
                    qe.push({i, j}); // Add the coordinates of the cell to the queue
                    dis[i][j] = 0; // Set distance of cells containing 0 to 0
                }
            }
        }

        // Direction vectors for exploring neighbors (right, left, up, down)
        vector<pair<int, int>> dir = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};

        // Perform BFS
        while (!qe.empty()) {
            // Get the current cell from the queue
            auto [x, y] = qe.front();
            qe.pop();
            
            // Explore all neighbors
            for (auto& [dx, dy] : dir) {
                int newX = x + dx; // Calculate new row index
                int newY = y + dy; // Calculate new column index
                
                // Check if the neighbor is within bounds and unvisited
                if (newX >= 0 && newX < m && newY >= 0 && newY < n && dis[newX][newY] == -1) {
                    dis[newX][newY] = dis[x][y] + 1; // Update distance for the neighbor
                    qe.push({newX, newY}); // Add the neighbor to the queue for further exploration
                }
            }
        }

        // Return the distance matrix
        return dis;
    }
};
```

---

### Explanation of Key Parts:

1. **Initialization**:
   - We loop through the input matrix to identify all cells with a `0`. These cells act as the starting points for BFS.

2. **Direction Control**:
   - The `dir` vector simplifies movement to neighboring cells. Each pair represents a change in row (`dx`) and column (`dy`).

3. **BFS**:
   - BFS ensures that the shortest distance is calculated first. Each layer corresponds to cells that are `1` step further from the `0` cells.

4. **Condition Check**:
   - The check `dis[newX][newY] == -1` ensures we only process unvisited cells. This prevents overwriting distances.

---

### Complexity:

1. **Time Complexity**:
   - Each cell is processed at most once. Total operations = \(O(m \times n)\), where \(m\) is the number of rows and \(n\) is the number of columns.

2. **Space Complexity**:
   - The space required for the `dis` matrix is \(O(m \times n)\). The queue can also hold at most \(O(m \times n)\) elements in the worst case.

---

### Example Walkthrough:

#### Input:
```plaintext
mat = [[0, 0, 0],
       [0, 1, 0],
       [1, 1, 1]]
```

#### Output:
```plaintext
[[0, 0, 0],
 [0, 1, 0],
 [1, 2, 1]]
```

- BFS starts from cells with `0` and propagates distances layer by layer to the `1` cells.

