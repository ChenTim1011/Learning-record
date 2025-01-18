[Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/description/)


```cpp
const int di[4] = {0, 0, 1, -1}; // Direction arrays for row movement: right, left, down, up
const int dj[4] = {1, -1, 0, 0}; // Direction arrays for column movement: right, left, down, up

class Solution {
public:
    // Function to check if a given cell (i, j) is outside the grid boundaries
    static inline bool isOutside(int i, int j, int r, int c) {
        return i < 0 || i >= r || j < 0 || j >= c;
    }

    // Helper function to pack distance (d), row index (i), and column index (j) into a single integer
    static inline unsigned pack(unsigned d, unsigned i, unsigned j) {
        return (d << 16) + (i << 8) + j; // Store distance in the highest bits, row index in the middle, and column index in the lowest bits
    }

    // Helper function to unpack a packed integer back into distance, row index, and column index
    static inline array<int, 3> unpack(unsigned info) {
        array<int, 3> ans;
        ans[0] = info >> 16;         // Extract distance (highest 16 bits)
        ans[1] = (info >> 8) & 255; // Extract row index (next 8 bits)
        ans[2] = info & 255;        // Extract column index (lowest 8 bits)
        return ans;
    }

    // Compute a unique index for a cell (i, j) based on its row and column
    static unsigned int idx(int i, int j, int c) {
        return i * c + j; // Flatten 2D coordinates into a single index
    }

    // Main function to calculate the minimum cost
    static int minCost(vector<vector<int>>& grid) {
        const int r = grid.size();  // Number of rows
        const int c = grid[0].size(); // Number of columns
        
        // Priority queue to implement Dijkstra's algorithm (min-heap based on distance)
        priority_queue<unsigned, vector<unsigned>, greater<>> pq;

        // Array to store the minimum cost to reach each cell
        unsigned* dist = (unsigned*)alloca(r * c * sizeof(unsigned));
        bitset<10000> viz = 0; // Bitset to track visited cells (grid size <= 10,000)

        fill(dist, dist + r * c, UINT_MAX); // Initialize distances to infinity
        pq.push(pack(0, 0, 0)); // Start from the top-left cell with cost 0
        dist[0] = 0;            // Distance to the starting cell is 0
        viz[0] = 1;             // Mark the starting cell as visited

        // Process cells until the priority queue is empty
        while (!pq.empty()) {
            auto info = pq.top(); // Get the cell with the smallest cost
            pq.pop();
            auto [d, i, j] = unpack(info); // Unpack distance, row, and column

            viz[idx(i, j, c)] = 1; // Mark the current cell as visited

            // If we reach the bottom-right cell, return the cost
            if (i == r - 1 && j == c - 1)
                return d;

            int x = grid[i][j]; // Current cell's direction

            // Iterate through all possible directions (right, left, down, up)
            for (int a = 0; a < 4; a++) {
                int s = i + di[a]; // Calculate the next row
                int t = j + dj[a]; // Calculate the next column

                // Skip invalid or already visited cells
                if (isOutside(s, t, r, c) || viz[idx(s, t, c)])
                    continue;

                // Calculate the new distance to the next cell
                // Cost is 0 if the current direction matches the intended direction, otherwise cost is 1
                int new_d = d + 1 - (a + 1 == x);

                // Flattened index of the next cell
                int b = idx(s, t, c);

                // Update the distance if a shorter path is found
                if (new_d < dist[b]) {
                    dist[b] = new_d;
                    pq.push(pack(new_d, s, t)); // Push the next cell into the priority queue
                }
            }
        }

        return INT_MAX; // If no valid path exists (should not occur in a valid input)
    }
};
```

---

### Explanation of the Algorithm

This problem is essentially finding the minimum cost to traverse a grid from the top-left corner to the bottom-right corner, where the grid has predefined directions, and changing a direction costs `1`.

The algorithm uses **Dijkstra's shortest path algorithm** with a priority queue to minimize the cost of traversal. Below are the detailed steps:

1. **Grid Representation**:
   - Each cell in the grid has a predefined direction (`1`, `2`, `3`, or `4`) corresponding to moving right, left, down, or up.
   - If the path follows the predefined direction, there is no cost to move.
   - If the path deviates from the predefined direction, it incurs a cost of `1`.

2. **Priority Queue**:
   - A priority queue is used to always process the cell with the smallest cost first.
   - Each entry in the queue is a packed integer containing the distance (cost), row index, and column index.

3. **Relaxation**:
   - For each cell, check all four possible directions (right, left, down, up).
   - Calculate the new cost to move to the next cell.
   - Update the distance if a shorter path is found and push the next cell into the priority queue.

4. **Stopping Condition**:
   - As soon as the algorithm processes the bottom-right cell, it returns the minimum cost to reach it.

5. **Efficiency**:
   - The algorithm processes each cell at most once due to the visited set (`viz`).
   - The priority queue ensures that cells are processed in increasing order of cost.
   - The time complexity is \(O((m \cdot n) \log(m \cdot n))\), where \(m\) and \(n\) are the dimensions of the grid.

This approach ensures that the algorithm finds the minimum cost to make the grid have at least one valid path.