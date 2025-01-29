[Redundant Connection](https://leetcode.com/problems/redundant-connection/description/)

### Union-Find (Disjoint Set Union, DSU) Detailed Tutorial

#### 1. **Basic Concepts**
**Union-Find** (also called Disjoint Set Union, DSU) is a data structure that manages elements grouped into disjoint sets. It supports two primary operations:
- **Find**: Determine which set an element belongs to (by finding its root).
- **Union**: Merge two sets containing elements `x` and `y`.

**Use Cases**:  
- Detecting cycles in graphs.
- Counting connected components (e.g., islands in grids).
- Solving dynamic connectivity problems (e.g., network connections).

---

#### 2. **Core Ideas**
- **Root Representation**: Each element points to its parent. The root of a set points to itself.
- **Path Compression**: Flatten the tree during `Find` operations to speed up future queries.
- **Union by Rank**: Merge smaller trees into larger ones to keep the tree balanced.

---

#### 3. **Code Walkthrough (with Annotations)**

```cpp
class UnionFind {
    vector<int> root; // Stores the parent/root of each element
    vector<int> rank; // Stores the "height" of the tree (for balancing)
public:
    UnionFind(int n) : root(n), rank(n) {
        rank.assign(n, 1); // Initialize all ranks to 1
        iota(root.begin(), root.end(), 0); // root[i] = i (each element is its own root initially)
    }

    // Find the root of x with path compression
    int Find(int x) {
        if (x == root[x]) return x; // Root found
        return root[x] = Find(root[x]); // Path compression: directly link x to the root
    }

    // Union the sets containing x and y
    bool Union(int x, int y) {
        int rootX = Find(x), rootY = Find(y);
        if (rootX == rootY) return false; // Already in the same set
        
        // Union by rank: attach the shorter tree to the taller one
        if (rank[rootX] > rank[rootY]) swap(rootX, rootY);
        root[rootX] = rootY; // Merge rootX into rootY
        
        // Update rank if the trees were equally tall
        if (rank[rootX] == rank[rootY]) rank[rootY]++;
        return true; // Successfully merged two sets
    }
};
```

---

#### 4. **Step-by-Step Example**

**Initialization**: Elements `0,1,2,3,4` (each is its own root).  
```
root = [0,1,2,3,4]
rank = [1,1,1,1,1]
```

**Operation 1**: `Union(1, 2)`  
- `Find(1) = 1`, `Find(2) = 2` → Different roots.  
- Merge `2` into `1` (since ranks are equal, `rank[1]` becomes 2).  
```
root = [0,1,1,3,4]
rank = [1,2,1,1,1]
```

**Operation 2**: `Union(2, 3)`  
- `Find(2) = 1`, `Find(3) = 3` → Different roots.  
- Merge `3` into `1` (since `rank[1] = 2 > rank[3] = 1`).  
```
root = [0,1,1,1,4]
rank = [1,2,1,1,1]
```

**Operation 3**: `Find(3)` → Compresses path to root `1`.  
```
root = [0,1,1,1,4] // 3 now directly points to 1
```

---

#### 5. **Solving Problems: Redundant Connection**

**Problem**: Given an undirected graph’s edge list, find the last edge that creates a cycle.  

**Solution**:  
- Use Union-Find to detect the first edge that connects already-connected nodes.  

```cpp
class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        const int n = edges.size();
        UnionFind G(n + 1); // Nodes are 1-indexed
        for (auto& e : edges) {
            if (!G.Union(e[0], e[1])) return e; // Cycle detected
        }
        return {};
    }
};
```

---

#### 6. **Complexity Analysis**
- **Time Complexity**: Nearly O(n) due to path compression and union by rank (amortized O(α(n)), where α is the inverse Ackermann function).  
- **Space Complexity**: O(n) for storing `root` and `rank` arrays.  

---

#### 7. **Key Applications**
1. **Cycle Detection**: Identify redundant edges in graphs.
2. **Connected Components**: Count islands in grids ([LeetCode 2658](https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/)).
3. **Graph Connectivity**: Check if a path exists between nodes ([LeetCode 1971](https://leetcode.com/problems/find-if-path-exists-in-graph/)).
4. **Advanced Grid Problems**: Split regions using slashes ([LeetCode 959](https://leetcode.com/problems/regions-cut-by-slashes/)).

---

#### 8. **Why Union-Find?**
- **Efficiency**: Path compression and union by rank make operations nearly constant time.  
- **Simplicity**: Easy to implement for dynamic connectivity problems.  
- **Versatility**: Adaptable to grids, graphs, and other structures.  

---

#### 9. **Common Pitfalls**
- **Indexing Errors**: Ensure nodes are 0- or 1-indexed consistently.  
- **Forgetting Path Compression**: Always implement `Find` with path compression.  
- **Ignoring Rank**: Union by rank is critical for balancing trees.  

