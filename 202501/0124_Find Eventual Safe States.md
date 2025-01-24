[Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/description/?envType=daily-question&envId=2025-01-24)

### Problem Description

In this problem, you are given a directed graph with `n` nodes, numbered from `0` to `n-1`. The graph is represented as a 2D integer array `graph`, where `graph[i]` is an integer array representing the nodes that node `i` has edges pointing to.

The goal is to find all the **safe nodes**.  
A **safe node** is defined as a node from which every possible path leads to a terminal node (a node with no outgoing edges) or other safe nodes.

---

### Solution Approach

We use **topological sorting** (Kahn's Algorithm) to solve this problem. Here's a detailed step-by-step approach:

---

### Step 1: Initialization

- Declare an array `indeg` to store the **in-degree** (number of incoming edges) for each node.
- Construct a **reverse adjacency list** (`adj`) to reverse the direction of the original graph. This will help us process nodes in reverse order.

---

### Step 2: Compute In-degree and Build Reverse Adjacency List

Traverse the entire `graph`, and for each node `i`:
- Add the edges pointing from its neighbors back to `i` in the reverse adjacency list `adj`.
- Increase the in-degree of node `i` for each incoming edge.

---

### Step 3: Initialize the Queue

- Add all nodes with in-degree `0` (terminal nodes) to a queue `q`. These nodes are inherently safe.

---

### Step 4: BFS Processing

Using BFS, process the nodes:
1. Pop a node `x` from the queue and mark it as **safe**.
2. For each neighbor `y` of node `x` in the reverse adjacency list `adj[x]`:
   - Decrement the in-degree of `y`.
   - If the in-degree of `y` becomes `0`, add it to the queue because all its dependencies are now safe.

---

### Step 5: Construct the Answer

- Traverse all nodes, and for each node marked as safe, add it to the result.

---

### Complexity Analysis

- **Time Complexity**: \(O(V + E)\), where \(V\) is the number of nodes and \(E\) is the number of edges. Each node and edge is processed once.
- **Space Complexity**: \(O(V + E)\), for the in-degree array, adjacency list, and queue.

---

### Step-by-Step Explanation with Example

#### Code:
```cpp
class Solution {
public:
    static vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        const int n = graph.size(); // Number of nodes
        vector<int> indeg(n, 0); // Initialize in-degree array
        vector<vector<int>> adj(n); // Reverse adjacency list

        // Step 1: Compute in-degree and build reverse adjacency list
        for (int i = 0; i < n; i++) {
            for (int j = graph[i].size() - 1; j >= 0; j--) {
                int v = graph[i][j]; // Node i points to node v
                adj[v].push_back(i); // Reverse the edge
                indeg[i]++; // Increase in-degree of i
            }
        }

        queue<int> q; // BFS queue

        // Step 2: Add all terminal nodes (in-degree 0) to the queue
        for (int i = n - 1; i >= 0; i--) {
            if (indeg[i] == 0) q.push(i);
        }

        vector<bool> safe(n, 0); // Mark safe nodes
        while (!q.empty()) {
            int x = q.front(); // Get the next node
            q.pop();
            safe[x] = 1; // Mark x as safe
            for (int y : adj[x]) { // For each neighbor of x in the reversed graph
                if (--indeg[y] == 0) // Decrease in-degree and check if it becomes 0
                    q.push(y); // Add y to the queue
            }
        }

        // Step 3: Collect the result
        vector<int> ans;
        for (int i = 0; i < n; i++) {
            if (safe[i]) ans.push_back(i); // Add safe nodes to the result
        }
        return ans;
    }
};
```

---

### Test Cases

#### Example 1:
**Input**:  
`graph = [[1,2],[2,3],[5],[0],[5],[],[]]`  
**Output**:  
`[2,4,5,6]`  
**Explanation**:  
- Nodes 5 and 6 are terminal nodes.  
- Nodes 2 and 4 lead only to terminal nodes.  

#### Example 2:
**Input**:  
`graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]`  
**Output**:  
`[4]`  
**Explanation**:  
- Only node 4 is a terminal node.  

---

### Key Intermediate Steps (Example 1)

#### **Initial State**:
- `indeg = [2, 2, 1, 1, 1, 0, 0]`
- `adj = [[], [0], [0, 1], [1], [], [2, 4], []]`
- Queue `q = [5, 6]` (terminal nodes).

#### **Step-by-Step BFS**:
1. Process node `5`:
   - Mark as safe: `safe[5] = 1`.
   - Reduce in-degrees of nodes `2` and `4`.
   - Updated `indeg = [2, 2, 0, 1, 0, 0, 0]`.
   - Add `2` and `4` to the queue.

2. Process node `6`:
   - Mark as safe: `safe[6] = 1`.

3. Process node `2`:
   - Mark as safe: `safe[2] = 1`.
   - Reduce in-degrees of nodes `0` and `1`.

4. Process node `4`:
   - Mark as safe: `safe[4] = 1`.

#### **Final Safe Nodes**:
- Nodes `2, 4, 5, 6`.

---

### Summary

This approach uses reverse adjacency lists and topological sorting to efficiently determine all safe nodes, ensuring correctness and optimal performance.