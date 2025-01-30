[Divide Nodes Into the Maximum Number of Groups](https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/description/)

### Problem Description

You are given a positive integer `n` representing the number of nodes in an undirected graph. The nodes are labeled from `1` to `n`.

You are also given a 2D integer array `edges`, where `edges[i] = [a_i, b_i]` indicates that there is a bidirectional edge between nodes `a_i` and `b_i`. Notice that the given graph may be disconnected.

Divide the nodes of the graph into `m` groups (1-indexed) such that:
1. Each node in the graph belongs to exactly one group.
2. For every pair of nodes in the graph that are connected by an edge \([a_i, b_i]\), if \(a_i\) belongs to group \(x\) and \(b_i\) belongs to group \(y\), then \(|y - x| = 1\).

Return the maximum number of groups (i.e., maximum `m`) into which you can divide the nodes. Return `-1` if it is impossible to group the nodes with the given conditions.

---

### Example 1

**Input:**
```text
n = 6, edges = [[1,2],[1,4],[1,5],[2,6],[2,3],[4,6]]
```

**Output:** 
```text
4
```

**Explanation:**  
The nodes can be grouped as follows:
- Add node 5 to group 1.
- Add node 1 to group 2.
- Add nodes 2 and 4 to group 3.
- Add nodes 3 and 6 to group 4.  

It can be shown that this is the maximum number of groups. Any attempt to create a fifth group will result in at least one edge violating the conditions.

---

### Example 2

**Input:**
```text
n = 3, edges = [[1,2],[2,3],[3,1]]
```

**Output:** 
```text
-1
```

**Explanation:**  
If we try to add node 1 to group 1, node 2 to group 2, and node 3 to group 3 to satisfy the first two edges, the third edge \([3,1]\) will fail because \(|3 - 1| = 2 \neq 1\).  
It can be shown that no grouping is possible for this graph.

---

### Constraints
- \(1 \leq n \leq 500\)
- \(1 \leq edges.length \leq 10^4\)
- \(edges[i].length = 2\)
- \(1 \leq a_i, b_i \leq n\)
- \(a_i \neq b_i\)
- There is at most one edge between any pair of vertices.

---

### Approach: **Method 2 - BFS + Multi-source Search**

#### Key Idea:
This approach involves simulating the grouping process by performing **Breadth-First Search (BFS)** starting from every node. For each node:
1. We calculate the maximum depth of the connected component (i.e., the maximum "group" index it can contribute to) using BFS.
2. At the same time, we ensure that the graph satisfies the bipartite property (no two adjacent nodes can belong to the same group).

If any node violates the bipartite property, the grouping is impossible, and we return `-1`. Otherwise, we sum up the maximum depths from all connected components.

---

#### Steps:

1. **Graph Representation:**  
   Use an adjacency list to store the graph. Each node \(i\) points to a list of its neighbors.

2. **BFS Traversal for Maximum Depth:**  
   - For each node \(i\), perform BFS starting from that node.  
   - Maintain a `visited` array to track the group index of each node. Initialize the starting node \(i\) with a group index of `1`.
   - As BFS progresses, increment the depth for each level, effectively grouping nodes into increasing indices.
   - Track the maximum depth reached during this BFS.

3. **Bipartite Validation:**  
   During BFS, check the conditions for the bipartite graph:
   - For each node \(cur\), check its neighbors.
   - If a neighbor has already been visited and the group index difference between the current node and the neighbor is not `1`, return `-1` as the graph cannot be grouped.

4. **Summing Maximum Depths:**  
   The sum of the maximum depths obtained from all BFS traversals gives the total number of groups.

---

#### Complexity:

- **Time Complexity:**  
  \(O(n + e)\), where \(n\) is the number of nodes and \(e\) is the number of edges.  
  - Building the adjacency list takes \(O(e)\).
  - BFS traverses each node and edge exactly once, taking \(O(n + e)\).

- **Space Complexity:**  
  \(O(n + e)\), where:
  - The adjacency list requires \(O(n + e)\) space.
  - Additional structures like the `visited` array and BFS queue take \(O(n)\) space.

---

#### Code:

```cpp
class Solution {
public:
    int magnificentSets(int n, vector<vector<int>>& edges) {
        // Step 1: Build the graph using adjacency list
        vector<vector<int>> graph(n + 1);
        for (auto& edge : edges) {
            graph[edge[0]].push_back(edge[1]);
            graph[edge[1]].push_back(edge[0]);
        }

        // Step 2: Initialize maxDepth array to store the maximum depth for each connected component
        vector<int> maxDepth(n + 1, 0);

        // Step 3: Perform BFS for each node
        for (int i = 1; i <= n; i++) {
            queue<int> q;
            vector<int> visited(n + 1, 0);
            q.push(i);
            visited[i] = 1; // Start with depth = 1

            int depth = 0;
            while (!q.empty()) {
                int size = q.size();
                depth++;

                for (int j = 0; j < size; j++) {
                    int cur = q.front();
                    q.pop();

                    // Traverse neighbors
                    for (int neighbor : graph[cur]) {
                        if (visited[neighbor] == 0) { // Unvisited node
                            visited[neighbor] = depth + 1;
                            q.push(neighbor);
                        } else if (abs(visited[neighbor] - visited[cur]) != 1) {
                            return -1; // Bipartite property violated
                        }
                    }
                }
            }

            // Update maxDepth for this connected component
            maxDepth[i] = max(maxDepth[i], depth);
        }

        // Step 4: Sum up the maximum depths for all connected components
        return accumulate(maxDepth.begin(), maxDepth.end(), 0);
    }
};
```

---

### Explanation of Code:

1. **Graph Construction:**  
   The graph is represented as an adjacency list where each node is mapped to its neighbors.

2. **BFS Traversal:**  
   For each node:
   - If the node is not visited, perform BFS to explore its connected component.
   - Track the depth of the traversal to determine the number of groups.
   - Ensure the bipartite condition is satisfied during BFS.

3. **Update Maximum Depths:**  
   After BFS, update the `maxDepth` array to reflect the maximum depth (groups) contributed by the current connected component.

4. **Final Result:**  
   Sum all the maximum depths from the `maxDepth` array to get the total number of groups. If any bipartite check fails, return `-1`.

---

This method is efficient and ensures both the correctness of the grouping and the maximum possible number of groups.