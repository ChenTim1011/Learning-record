[Making A Large Island](https://leetcode.com/problems/making-a-large-island/description/?envType=daily-question&envId=2025-01-31)


### Problem Explanation

You are given a binary matrix `grid` of size \( n \times n \), where:
- `1` represents land.
- `0` represents water.

You are allowed to change at most one `0` to `1`. After doing so, you must determine the size of the largest possible island. 

An **island** is defined as a group of connected `1`s, where connectivity is limited to 4 directions (up, down, left, right).

---

### Key Insights for the Solution
1. **Union-Find Data Structure**:
   - We can use Union-Find to group connected `1`s into "islands" and efficiently find their size.
   - Each connected component (island) will have a unique "root," and the size of the island is tracked using a `Size` array.

2. **Steps in the Solution**:
   - **Step 1: Identify connected components**:
     Traverse the grid and group all `1`s into connected components using Union-Find. Calculate the size of each component.
   - **Step 2: Consider flipping each `0`**:
     For each `0`, check the islands it could connect to by flipping itself into a `1`. Calculate the combined size of these connected components.
   - **Step 3: Track the maximum size**:
     Compare the resulting island size after flipping with the current maximum size.

3. **Edge Cases**:
   - If the grid is entirely land (`1`s), the largest island is the total number of cells in the grid.
   - If the grid is entirely water (`0`s), the largest island will have size \(1\) after flipping one `0` to `1`.

---

### Algorithm in Detail

#### Step 1: Union-Find Class Definition
Union-Find is used to group connected `1`s into components.

- **Find(x)**: Returns the root of `x`. Uses path compression for efficiency.
- **Union(x, y)**: Merges the components of `x` and `y`. Merges smaller components into larger ones for size optimization.
- **Size Tracking**: Tracks the size of each connected component.

#### Step 2: Traverse the Grid and Create Islands
- For every cell in the grid that is `1`, check its neighbors (down and right) and merge them into the same component using the `Union` operation.
- Update the size of each component in the `Size` array.

#### Step 3: Check All Possible Flips
- For every `0` in the grid:
  1. Check its four neighbors (up, down, left, right).
  2. Use `Find` to get the root of each unique neighboring component.
  3. Calculate the combined size of the connected components by summing their sizes (avoiding duplicates).
  4. Update the maximum size of the island.

---

### Code Implementation in C++

```cpp
class UnionFind {
public:
    vector<int> root, size;
    
    UnionFind(int n) : root(n), size(n, 1) {
        iota(root.begin(), root.end(), 0); // Initialize roots
    }
    
    int Find(int x) {
        return (x == root[x]) ? x : root[x] = Find(root[x]); // Path compression
    }
    
    bool Union(int x, int y) {
        int rootX = Find(x), rootY = Find(y);
        if (rootX == rootY) return false;
        
        if (size[rootX] > size[rootY]) {
            size[rootX] += size[rootY];
            root[rootY] = rootX;
        } else {
            size[rootY] += size[rootX];
            root[rootX] = rootY;
        }
        return true;
    }
};

class Solution {
public:
    int largestIsland(vector<vector<int>>& grid) {
        int n = grid.size();
        UnionFind uf(n * n); // Initialize Union-Find for a grid of size n * n
        vector<vector<int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int maxIsland = 0;
        
        // Step 1: Connect all land cells (1s) to form components
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    int idx = i * n + j;
                    for (auto dir : directions) {
                        int x = i + dir[0], y = j + dir[1];
                        if (x >= 0 && x < n && y >= 0 && y < n && grid[x][y] == 1) {
                            uf.Union(idx, x * n + y);
                        }
                    }
                }
            }
        }
        
        // Step 2: Calculate initial maximum island size
        for (int i = 0; i < n * n; i++) {
            if (grid[i / n][i % n] == 1) {
                maxIsland = max(maxIsland, uf.size[uf.Find(i)]);
            }
        }
        
        // Step 3: Try flipping each 0 and calculate the new island size
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    unordered_set<int> uniqueComponents;
                    int newSize = 1; // Flipping this 0 adds itself
                    
                    // Check all neighbors
                    for (auto dir : directions) {
                        int x = i + dir[0], y = j + dir[1];
                        if (x >= 0 && x < n && y >= 0 && y < n && grid[x][y] == 1) {
                            int root = uf.Find(x * n + y);
                            if (uniqueComponents.insert(root).second) {
                                newSize += uf.size[root];
                            }
                        }
                    }
                    
                    maxIsland = max(maxIsland, newSize);
                }
            }
        }
        
        return maxIsland;
    }
};
```

---

### Complexity Analysis
1. **Time Complexity**:
   - **Union-Find Operations**: \( O(n^2 \cdot \alpha(n^2)) \), where \(\alpha\) is the inverse Ackermann function.
   - **Flipping Operation**: \( O(n^2 \cdot 4) \), as each `0` checks at most 4 neighbors.
   - Overall: \( O(n^2) \), as \(\alpha(n^2)\) is almost constant for practical purposes.

2. **Space Complexity**:
   - Union-Find storage for `root` and `size`: \( O(n^2) \).
   - Overall: \( O(n^2) \).

---

### Example Walkthrough
#### Example 1:
Input:
```
grid = [[1, 0],
        [0, 1]]
```
- Initial Union-Find groups: Two islands of size 1 each.
- Flip `grid[0][1]` or `grid[1][0]` to connect the two islands.
- Largest island size: \( 3 \).

Output: `3`.

#### Example 2:
Input:
```
grid = [[1, 1],
        [1, 0]]
```
- Initial Union-Find groups: One island of size \( 3 \).
- Flip `grid[1][1]` to add the `0`.
- Largest island size: \( 4 \).

Output: `4`.