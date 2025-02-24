[Most Profitable Path in a Tree](https://leetcode.com/problems/most-profitable-path-in-a-tree/description/)

## **📌 Problem Explanation**
### **1️⃣ Problem Statement**
- We are given an **undirected tree** with `n` nodes labeled from `0` to `n-1`.
- The tree is described using `edges`, where `edges[i] = [ai, bi]` represents an edge between nodes `ai` and `bi`.
- **Alice starts at node `0` and moves toward a leaf node** in any path she chooses.
- **Bob starts at node `bob` and moves towards node `0`** in a predetermined way.
- Each node `i` has a monetary `amount[i]` associated with it:
  - If `amount[i]` is **positive**, Alice can **earn** this amount when passing through the node.
  - If `amount[i]` is **negative**, Alice must **pay** to pass through the node.

### **2️⃣ Game Rules**
- **Alice moves towards some leaf node** using an optimal strategy to maximize her profit.
- **Bob moves towards node `0`** along the shortest path.
- **Alice and Bob affect the earnings at each node:**
  1. **If Alice arrives at a node before Bob**, she **receives the full amount**.
  2. **If Bob arrives at a node before Alice**, Alice **gets nothing**.
  3. **If Alice and Bob arrive at a node at the same time**, they **split the amount equally**.
- **Alice stops moving when she reaches a leaf node**.
- **Bob stops moving when he reaches node `0`**.

### **3️⃣ Example Walkthrough**
#### **Example 1:**
```cpp
Input: edges = [[0,1],[1,2],[1,3],[3,4]], bob = 3, amount = [-2,4,2,-4,6]
Output: 6
```
#### **Step-by-step breakdown**
1. **Construct the tree from edges:**
   ```
       0
       |
       1
      / \
     2   3
         |
         4
   ```
2. **Alice starts at `0`, Bob starts at `3`.**
3. **Bob's movement:** `3 → 1 → 0`
4. **Alice's movement (choosing the best path to maximize profit):**  
   - **Alice moves `0 → 1`**, Bob also reaches `1` at the same time. **Alice earns `4 / 2 = 2`**.
   - **Alice moves `1 → 3`**, but Bob has already passed through, so Alice gets nothing here.
   - **Alice moves `3 → 4`**, earning `6`.

   **Total profit: `-2 + 2 + 0 + 6 = 6`.** ✅

---

## **🛠 Solution Approach**
This problem can be efficiently solved using **two DFS traversals**:

1. **First DFS**:  
   - Find the **parent** of each node to construct the tree structure.
2. **Compute Bob’s arrival times**:  
   - Store the **earliest time Bob reaches each node** using a simple traversal.
3. **Second DFS**:  
   - Compute the **maximum possible profit Alice can obtain** by navigating optimally.

---

## **📌 Detailed Implementation**
### **Step 1️⃣: Construct the Tree Using an Adjacency List**
Since the tree is undirected, we represent it as an **adjacency list**.

```cpp
for (int i = 0; i < n; i++) adj[i].clear();

for (auto& e : edges) {
    int u = e[0], v = e[1];
    adj[u].push_back(v);
    adj[v].push_back(u);
}
```
This converts the `edges` array into a **graph structure** for easy traversal.

---

### **Step 2️⃣: First DFS - Finding Parent Nodes**
We perform a **DFS traversal** to find the **parent of each node**, which helps us track **Bob’s path to node `0`**.

```cpp
void dfs(int i, int p) {
    parent[i] = p;
    for (int j : adj[i]) {
        if (j == p) continue;  // Avoid going back to parent
        dfs(j, i);
    }
}
```
- This stores **`parent[i]` as the parent of node `i`**.
- It helps us later when computing **Bob's movement back to `0`**.

---

### **Step 3️⃣: Compute Bob’s Arrival Times**
- **Bob moves from `bob` back to `0`**, and we track how many steps he takes to reach each node.

```cpp
fill(Bob, Bob + n, INT_MAX);  // Initialize Bob's arrival times to infinity

for (int x = bob, move = 0; x != -1; x = parent[x]) {
    Bob[x] = move++;  // Bob reaches node `x` in `move` steps
}
```
- **Bob starts at `bob` and follows `parent[]` back to `0`**.
- The array `Bob[i]` now contains **the number of steps Bob needs to reach each node `i`**.

---

### **Step 4️⃣: Second DFS - Compute Alice’s Maximum Profit**
We now let Alice traverse the tree to maximize her earnings.

```cpp
int dfs_sum(int i, int dist, int prev, vector<int>& amount) {
    int alice = 0;

    // Determine Alice's earnings at this node
    if (dist < Bob[i]) alice = amount[i];   // Alice arrives first, takes full amount
    else if (dist == Bob[i]) alice = amount[i] / 2;  // Alice and Bob arrive at the same time, split amount

    bool isLeaf = true;  // Assume it's a leaf node initially
    int maxLeafSum = INT_MIN;  // Track the max sum from any child

    for (int j : adj[i]) {
        if (j == prev) continue;  // Avoid revisiting the parent
        isLeaf = false;  // If a child exists, it's not a leaf node
        maxLeafSum = max(maxLeafSum, dfs_sum(j, dist + 1, i, amount));
    }

    return alice + (isLeaf ? 0 : maxLeafSum);
}
```
- **Alice's earnings at each node depend on when she arrives relative to Bob.**
- **If Alice reaches before Bob**, she gets the **full `amount[i]`**.
- **If Alice reaches at the same time as Bob**, she gets **half of `amount[i]`**.
- **Alice recursively explores all possible paths to find the maximum profit.**

---

### **Final Function: Solving the Problem**
```cpp
int mostProfitablePath(vector<vector<int>>& edges, int bob, vector<int>& amount) {
    const int n = edges.size() + 1;

    for (int i = 0; i < n; i++) adj[i].clear();

    for (auto& e : edges) {
        int u = e[0], v = e[1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    dfs(0, -1);

    // Compute Bob's reach time
    fill(Bob, Bob + n, INT_MAX);
    for (int x = bob, move = 0; x != -1; x = parent[x]) {
        Bob[x] = move++;
    }

    return dfs_sum(0, 0, -1, amount);
}
```
- **First, construct the adjacency list.**
- **Find `parent[]` using DFS.**
- **Compute Bob’s arrival times.**
- **Use DFS to find the most profitable path for Alice.**

---

## **Time Complexity Analysis**
- **DFS for Parent Calculation** → \( O(n) \)
- **Compute Bob’s Arrival Times** → \( O(n) \)
- **DFS for Maximum Profit** → \( O(n) \)
- **Total Complexity**: **\( O(n) \)**, efficient for \( n \leq 10^5 \).

---

## **✅ Summary**
- **First DFS**: Find parent nodes.
- **Compute Bob’s movement**: Track when Bob reaches each node.
- **Second DFS**: Let Alice navigate optimally for maximum earnings.
- **Time Complexity**: \( O(n) \), optimal for large inputs.
