[Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/description/)


### Problem Restatement
You’re given a 2D height map (`heightMap`) where each cell represents the elevation at that point. The task is to calculate how much water can be trapped after raining, considering the walls formed by higher elevations.

---

### Key Intuition

The core idea is that **water can only be trapped at an inner cell if it is surrounded by taller walls.** This means:
1. The water level at a cell is determined by the shortest wall (or boundary) surrounding it.
2. If a cell's height is less than this shortest wall, water is trapped at this cell. The amount of water trapped is the difference between the wall height and the cell's height.

To compute this in a 2D grid:
- Start with the **boundary cells**, as no water can be trapped at the edges.
- Use a **priority queue (min-heap)** to always process the cell with the **lowest height** first. This ensures that we handle potential water flow correctly because water can only flow from high to low elevations.

---

### Approach and Algorithm

#### Step 1: Initialize
1. **Visited Array**: Use a 2D array `visited` to mark which cells have been processed.
2. **Min-Heap (Priority Queue)**: This stores cells as `(height, (row, col))`. The heap always processes the cell with the smallest height first.
3. **Boundary Initialization**: Add all boundary cells (first row, last row, first column, last column) to the heap and mark them as visited.

#### Step 2: Process Cells Using BFS
1. Pop the cell with the smallest height from the heap.
2. For each of its four neighbors (up, down, left, right):
   - If the neighbor is **not visited** and **within bounds**, compute the trapped water at the neighbor.
   - Trapped water is `max(0, current_height - neighbor_height)`.
   - Update the neighbor's height in the heap to `max(current_height, neighbor_height)` (since it becomes part of the new boundary).
   - Mark the neighbor as visited.

#### Step 3: Repeat Until Heap is Empty
- Continue processing cells until the heap is empty.
- The sum of trapped water for all cells gives the final result.

---

### Code Walkthrough

#### Step 1: Initialize Variables and Add Boundary Cells
```cpp
int m = heightMap.size();       // Number of rows
int n = heightMap[0].size();    // Number of columns
vector<vector<bool>> visited(m, vector<bool>(n, false)); // Visited array

priority_queue<pair<int, pair<int, int>>, 
               vector<pair<int, pair<int, int>>>, 
               greater<pair<int, pair<int, int>>>> pq; // Min-heap

// Add boundary cells to the heap and mark them as visited
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
            pq.push({heightMap[i][j], {i, j}});
            visited[i][j] = true;
        }
    }
}
```

- **Purpose**: Add all boundary cells to the heap. These cells act as the initial container for the inner cells.
- **Complexity**: \(O(m + n)\) for the boundary cells.

#### Step 2: Process Cells Using Priority Queue
```cpp
vector<vector<int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}; // Directions for neighbors
int water = 0;  // Total trapped water

while (!pq.empty()) {
    auto p = pq.top();    // Get the cell with the smallest height
    pq.pop();

    int height = p.first;            // Height of the current cell
    int row = p.second.first;        // Row of the current cell
    int col = p.second.second;       // Column of the current cell

    // Explore all 4 directions
    for (auto& dir : directions) {
        int nextRow = row + dir[0];
        int nextCol = col + dir[1];

        // If the neighbor is within bounds and not visited
        if (nextRow >= 0 && nextRow < m && nextCol >= 0 && nextCol < n && !visited[nextRow][nextCol]) {
            // Water trapped is the difference between current height and neighbor's height
            water += max(0, height - heightMap[nextRow][nextCol]);

            // Add the neighbor to the heap with updated height
            pq.push({max(height, heightMap[nextRow][nextCol]), {nextRow, nextCol}});

            // Mark the neighbor as visited
            visited[nextRow][nextCol] = true;
        }
    }
    return water;
}
```

- **Purpose**: Process the lowest cell in the heap and compute the water trapped for its neighbors.
- **Key Logic**:
  - Use the **max of current height and neighbor height** to update the boundary for future processing.
  - Only unvisited neighbors are considered.
  - Trapped water at a neighbor is calculated as `max(0, current_height - neighbor_height)`.


---

### Example Walkthrough

#### Input:
```plaintext
heightMap = [
    [1, 4, 3, 1, 3, 2],
    [3, 2, 1, 3, 2, 4],
    [2, 3, 3, 2, 3, 1]
]
```

1. **Initialization**:
   - Boundary cells: \([1, 4, 3, 1, 3, 2], [3, 2, 4, ...]\)
   - Priority Queue: Min-heap stores boundary cells by height.
   - Start with the smallest height: \(1\).

2. **First Iteration**:
   - Process cell at \((0, 0)\) with height \(1\).
   - Explore its neighbors.
   - Neighbor at \((1, 0)\) has height \(3\). No water is trapped.

3. **Subsequent Iterations**:
   - Continue processing cells in the order of their heights, updating the boundary dynamically.
   - Calculate water trapped at each step.

#### Output:
\[
\text{Total water trapped} = 4
\]

---

### Complexity Analysis

1. **Time Complexity**:
   - \(O(m \times n \log(m \times n))\): Each cell is processed once, and heap operations take \(O(\log(m \times n))\).

2. **Space Complexity**:
   - \(O(m \times n)\): For the visited array and the priority queue.

---

### Summary
This approach leverages a **priority queue** to simulate water flow from the lowest boundary inward, ensuring that trapped water is calculated correctly for each cell. The algorithm is efficient and works for large grids due to its \(O(m \times n \log(m \times n))\) time complexity.