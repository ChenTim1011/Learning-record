[Maximum Employees to Be Invited to a Meeting](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/description/)


This problem is a graph theory problem where we need to determine the maximum number of employees that can be invited to a meeting, given the condition that each employee must be seated next to their favorite person. The key to solving this problem lies in identifying **chains** and **cycles** in the graph and combining them to maximize the number of invited employees.

---

### Problem Analysis
1. **Each employee has exactly one favorite person**:
   - This forms a **directed graph**, where each employee is a node, and there is a directed edge pointing to their favorite person.
2. **Possible Structures**:
   - **Chains**: Starting points are employees who are not anyone's favorite (nodes with an in-degree of 0).
   - **Cycles**: For example, Employee A likes Employee B, B likes Employee C, and C likes Employee A.
   - **Cycles of length 2**: For instance, Employee A likes Employee B, and B likes Employee A.

---

### Solution
The solution can be broken down into the following steps:

---

#### **Step 1: Calculate the in-degree of each node**
We use an `inDegree` array to track how many employees like each person.

- **Code Snippet**:
    ```cpp
    vector<int> inDegree(n, 0);
    for (int fav : favorites) {
        inDegree[fav]++;
    }
    ```

---

#### **Step 2: Identify Chains**
1. **Nodes with in-degree 0 are the starting points of chains**:
   - Add these nodes to a queue for topological sorting.
2. **Extend the chain**:
   - For each node in the queue, calculate the chain length and reduce the in-degree of the next node (the node they like).
   - If the in-degree of the next node becomes 0, add it to the queue.

- **Code Snippet**:
    ```cpp
    queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        visited[node] = true;

        int next = favorites[node];
        chainLengths[next] = chainLengths[node] + 1;
        if (--inDegree[next] == 0) {
            q.push(next);
        }
    }
    ```

---

#### **Step 3: Detect Cycles**
1. For all unvisited nodes, these nodes must belong to cycles.
2. Traverse the cycle and calculate its length until returning to the starting node.

- **Code Snippet**:
    ```cpp
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            int current = i, cycleLength = 0;
            while (!visited[current]) {
                visited[current] = true;
                current = favorites[current];
                cycleLength++;
            }

            if (cycleLength == 2) {
                totalChains += 2 + chainLengths[i] + chainLengths[favorites[i]];
            } else {
                maxCycle = max(maxCycle, cycleLength);
            }
        }
    }
    ```

---

#### **Step 4: Handle Cycles of Length 2**
- If a cycle has a length of 2 (e.g., A and B like each other), additional employees can connect to this cycle through chains on both sides.
- Add these cases to the total chain length.

---

#### **Step 5: Return the Maximum Value**
Finally, return the maximum of the two possibilities:
- `maxCycle`: The size of the largest cycle.
- `totalChains`: Chains combined with cycles of length 2.

---

### Full Code
```cpp
class Solution {
public:
    int maximumInvitations(vector<int>& favorites) {
        int n = favorites.size();
        vector<int> inDegree(n, 0), chainLengths(n, 0);
        vector<bool> visited(n, false);

        // Calculate in-degrees for all nodes
        for (int fav : favorites) {
            inDegree[fav]++;
        }

        // Find chain starting points (nodes with in-degree 0)
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }

        // Calculate chain lengths
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            visited[node] = true;

            int next = favorites[node];
            chainLengths[next] = chainLengths[node] + 1;
            if (--inDegree[next] == 0) {
                q.push(next);
            }
        }

        // Handle cycles and special cases for cycles of length 2
        int maxCycle = 0, totalChains = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                int current = i, cycleLength = 0;
                while (!visited[current]) {
                    visited[current] = true;
                    current = favorites[current];
                    cycleLength++;
                }

                if (cycleLength == 2) {
                    totalChains += 2 + chainLengths[i] + chainLengths[favorites[i]];
                } else {
                    maxCycle = max(maxCycle, cycleLength);
                }
            }
        }

        // Return the maximum of maxCycle and totalChains
        return max(maxCycle, totalChains);
    }
};
```

---

### Summary of the Solution
The core idea of the solution is to process the **chains and cycles** in the graph:
1. **Chains** are processed using topological sorting.
2. **Cycles** are detected and measured by traversing unvisited nodes.
3. Special handling is applied for cycles of length 2 to combine chains on both sides.

The **time complexity** is **O(n)** because every node and edge is processed a constant number of times. The **space complexity** is also **O(n)** due to the arrays used for tracking in-degrees, chain lengths, and visited status. This makes the solution efficient and suitable for large inputs.