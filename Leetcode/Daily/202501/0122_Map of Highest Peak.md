[Map of Highest Peak](https://leetcode.com/problems/map-of-highest-peak/description/)


### **Problem Explanation**

#### **Problem Statement**
You are given an `m x n` binary matrix `isWater` representing a map of water (`1`) and land (`0`) cells. You must assign a height to every cell such that:
1. Water cells (`isWater[i][j] == 1`) have a height of `0`.
2. Adjacent cells (north, south, east, west) must have an absolute height difference of at most `1`.
3. The maximum height in the resulting matrix is maximized.

Return the matrix `height` where `height[i][j]` represents the height of cell `(i, j)`.

#### **Example**

1. **Input:** `isWater = [[0,1],[0,0]]`  
   **Output:** `[[1,0],[2,1]]`  
   Explanation:  
   - Water cell (blue) has height `0`.  
   - Land cells are assigned heights such that the difference between adjacent cells is `1`.

2. **Input:** `isWater = [[0,0,1],[1,0,0],[0,0,0]]`  
   **Output:** `[[1,1,0],[0,1,1],[1,2,2]]`  
   Explanation:  
   - The maximum height of the matrix is `2`.  

#### **Constraints**
- \( 1 \leq m, n \leq 1000 \)
- At least one water cell is present in the matrix.

---

### **Solution Explanation**

To solve this problem, we need to calculate the distance of each land cell to the nearest water cell. The farther a land cell is from water, the higher its height will be. BFS (Breadth-First Search) is the most suitable algorithm for this because it naturally expands layer by layer, which ensures that heights are assigned correctly.

#### **Steps to Solve**
1. **Initialization**:  
   - Create a matrix `ans` of size \( m \times n \), initialized to `-1` to represent unvisited cells.
   - Add all water cells to a queue and set their height to `0`.

2. **Breadth-First Search**:  
   - While the queue is not empty, pop a cell `(i, j)` and process its neighbors.
   - For each neighbor `(x, y)` that is within bounds and unvisited (`ans[x][y] == -1`), set its height to `ans[i][j] + 1` and add it to the queue.

3. **Return the Result**:  
   - After the BFS completes, `ans` contains the desired heights for all cells.

#### **Complexity**
1. **Time Complexity**:  
   - Each cell is visited exactly once, and for each cell, we check at most 4 neighbors.  
   - Total complexity: \( O(m \times n) \).

2. **Space Complexity**:  
   - The queue can hold at most \( O(m \times n) \) elements in the worst case.  
   - Total space complexity: \( O(m \times n) \).

---

### **Why BFS and Not DFS?**

1. **Layered Expansion**:  
   - BFS explores all cells at distance `1` before moving to cells at distance `2`. This ensures that each cell is assigned the smallest possible height based on its distance to water.  
   - DFS, on the other hand, explores deeply along one path before backtracking, making it unsuitable for distance-based problems.

2. **Efficiency**:  
   - In BFS, each cell is processed only once. In DFS, a cell might be visited multiple times during backtracking, leading to inefficiency.

3. **Implementation Simplicity**:  
   - BFS uses a queue, which makes it easier to manage visited cells in a distance-based manner. DFS would require additional logic to keep track of minimum distances.

4. **Stack Overflow in DFS**:  
   - For large grids (e.g., \( 1000 \times 1000 \)), the recursive depth of DFS might exceed the stack limit, causing stack overflow.

---

### **Conclusion**

- **BFS** is the optimal choice for this problem due to its layered traversal, ensuring correct height assignment while being efficient and simple to implement.  
- The solution guarantees a time complexity of \( O(m \times n) \), which is suitable for the problem's constraints.


### **Solution**
```cpp
class Solution {
 public:
  vector<vector<int>> highestPeak(vector<vector<int>>& isWater) {
    // Get the dimensions of the matrix
    int m = isWater.size();
    int n = isWater[0].size();
    
    // Initialize the answer matrix with -1 to denote unvisited cells
    vector<vector<int>> ans(m, vector<int>(n, -1));
    
    // Queue to perform BFS, storing water cells initially
    queue<pair<int, int>> q;

    // Add all water cells to the queue and set their height to 0
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        if (isWater[i][j] == 1) {
          q.push({i, j}); // Add water cell to the queue
          ans[i][j] = 0;  // Water cells have a height of 0
        }
      }
    }

    // Define directions for moving: right, down, left, up
    const vector<pair<int, int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    // BFS to assign heights to all cells
    while (!q.empty()) {
      auto [i, j] = q.front(); // Get the front cell in the queue
      q.pop();

      // Check all four directions
      for (auto& [dx, dy] : dirs) {
        int x = i + dx, y = j + dy;
        // If the neighboring cell is within bounds and unvisited
        if (x >= 0 && x < m && y >= 0 && y < n && ans[x][y] == -1) {
          ans[x][y] = ans[i][j] + 1; // Assign height as current height + 1
          q.push({x, y});           // Add the cell to the queue for further processing
        }
      }
    }

    return ans; // Return the height matrix
  }
};
```