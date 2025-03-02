Reference:
[奉勸大家把資工學科熟讀](https://www.dcard.tw/f/graduate_school/p/258129327)


## **1. What’s the Difference Between BFS and DFS? When to Use Them?**
- **BFS (Breadth-First Search)**  
  - Uses a **Queue**, exploring level by level  
  - Suitable for **shortest path problems (in unweighted graphs)**, such as finding the **minimum number of hops**  
  - **Time Complexity**: \( O(V + E) \), where \( V \) is the number of vertices and \( E \) is the number of edges  

- **DFS (Depth-First Search)**  
  - Uses a **Stack (or recursion)**, exploring as deep as possible before backtracking  
  - Suitable for **connectivity checks, cycle detection, and topological sorting**  
  - **Time Complexity**: \( O(V + E) \)  

**Comparison of Use Cases**:

| Problem Type | BFS | DFS |
|-------------|----|----|
| **Shortest Path (Unweighted Graph)** | ✅ | ❌ |
| **Maze Solving (Shortest Path)** | ✅ | ❌ |
| **Cycle Detection** | ✅ | ✅ |
| **Topological Sorting (DAG)** | ❌ | ✅ |

---

## **2. Give an Example of a Greedy Algorithm**
A classic greedy algorithm is the **Activity Selection Problem**.

**Problem Statement**:  
Given \( n \) activities, each with a start time \( s_i \) and end time \( f_i \), select the maximum number of activities that do not overlap.

**Greedy Strategy**:
1. **Sort activities by their ending time**  
2. **Always select the earliest finishing non-overlapping activity**  

**Time Complexity**: \( O(n \log n) \) (due to sorting)

---

## **3. What is Dynamic Programming? How Does It Reduce Complexity?**
- **Dynamic Programming (DP)** solves problems by breaking them into **overlapping subproblems** and **storing results** to avoid recomputation.  
- Suitable for **optimization problems** such as **shortest paths, longest subsequence, and knapsack problems**.  

### **Complexity Reduction Example**
- **Fibonacci Sequence**  
  - **Naïve recursion**: \( O(2^n) \)  
  - **DP (Memoization/Bottom-Up)**: \( O(n) \)  

- **0/1 Knapsack Problem**  
  - **Brute Force (Exhaustive Search)**: \( O(2^n) \)  
  - **DP (Table-based Memoization)**: \( O(nW) \), where \( W \) is the knapsack capacity  

---

## **4. Why Use Inheritance?**
- **Inheritance** allows **subclasses** to reuse functionality from **superclasses**, reducing code duplication.  
- **Use Cases**:
  1. **Code Reuse** – Extract common behaviors into a superclass  
  2. **Polymorphism** – Allows subclasses to override behavior  
  3. **Architecture Design** – Example: `Animal → Dog, Cat`, both implementing `speak()`  

---

## **5. When is Bubble Sort Best and Worst?**
- **Best Case**: When the array is **already sorted**, requiring only one pass (**\( O(n) \)**).  
- **Worst Case**: When the array is **in reverse order**, requiring full nested loops (**\( O(n^2) \)**).  

---

## **6. Sorting a Large Set of Numbers Using a Tree**
Sorting can be achieved using a **Binary Search Tree (BST) or a Balanced Tree (AVL, Red-Black Tree)**.

1. **Insert numbers into a BST**  
   - Each insertion takes \( O(\log n) \) (for a balanced tree)  
2. **Perform Inorder Traversal**  
   - Produces a sorted sequence  
   - **Time Complexity**: \( O(n \log n) \) (balanced tree)  
   - **Worst-case (unbalanced tree): \( O(n^2) \)**  

## **Sorting Large Numbers Using an AVL Tree (C++ Implementation)**

```cpp
#include <iostream>
using namespace std;

// Node structure
struct Node {
    int key;
    Node *left, *right;
    int height;
};

// Get node height
int getHeight(Node* node) {
    return node ? node->height : 0;
}

// Get balance factor
int getBalanceFactor(Node* node) {
    return node ? getHeight(node->left) - getHeight(node->right) : 0;
}

// Create new node
Node* newNode(int key) {
    Node* node = new Node();
    node->key = key;
    node->left = node->right = nullptr;
    node->height = 1;
    return node;
}

// Right Rotation (LL Case)
Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(getHeight(y->left), getHeight(y->right)) + 1;
    x->height = max(getHeight(x->left), getHeight(x->right)) + 1;

    return x;
}

// Left Rotation (RR Case)
Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(getHeight(x->left), getHeight(x->right)) + 1;
    y->height = max(getHeight(y->left), getHeight(y->right)) + 1;

    return y;
}

// Insert into AVL Tree
Node* insert(Node* node, int key) {
    if (!node) return newNode(key);

    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else
        return node;

    node->height = 1 + max(getHeight(node->left), getHeight(node->right));
    int balance = getBalanceFactor(node);

    // LL Case
    if (balance > 1 && key < node->left->key) return rightRotate(node);
    // RR Case
    if (balance < -1 && key > node->right->key) return leftRotate(node);
    // LR Case
    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }
    // RL Case
    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}

// Inorder Traversal (Sorted Output)
void inorder(Node* root) {
    if (root) {
        inorder(root->left);
        cout << root->key << " ";
        inorder(root->right);
    }
}

int main() {
    Node* root = nullptr;
    int arr[] = {10, 20, 30, 40, 50, 25};
    for (int x : arr) root = insert(root, x);

    cout << "Sorted numbers: ";
    inorder(root);
    return 0;
}
```

**Output**:
```
Sorted numbers: 10 20 25 30 40 50
```

This ensures an **\( O(n \log n) \) complexity sorting using AVL Tree**.


---

## **7. How to Detect a Cycle in a Graph? What’s the Complexity?**
- **Directed Graph**
  - **DFS + Visit Status (White-Gray-Black Method)**  
  - **Time Complexity**: \( O(V + E) \)  

- **Undirected Graph**
  - **DFS + Union-Find (Disjoint Set)**  
  - **Time Complexity**: \( O(\alpha(n)) \) (near constant time)  


### Graph Cycle Detection (DFS + Union-Find)

#### **Directed Graph - DFS with White-Gray-Black Marking**
```cpp
#include <iostream>
#include <vector>

using namespace std;

enum Color { WHITE, GRAY, BLACK };

bool dfs(vector<vector<int>>& graph, vector<Color>& color, int node) {
    color[node] = GRAY; // Mark as "being visited"

    for (int neighbor : graph[node]) {
        if (color[neighbor] == GRAY) return true;  // Cycle detected
        if (color[neighbor] == WHITE && dfs(graph, color, neighbor)) return true;
    }

    color[node] = BLACK; // Mark as "fully visited"
    return false;
}

bool hasCycle(vector<vector<int>>& graph, int V) {
    vector<Color> color(V, WHITE);

    for (int i = 0; i < V; i++) {
        if (color[i] == WHITE && dfs(graph, color, i)) return true;
    }
    return false;
}

int main() {
    int V = 4;
    vector<vector<int>> graph(V);
    
    graph[0] = {1};
    graph[1] = {2};
    graph[2] = {0};  // Cycle
    graph[3] = {1};

    cout << (hasCycle(graph, V) ? "Cycle found" : "No cycle") << endl;
    return 0;
}
```
**Output:**
```
Cycle found
```
This approach uses **DFS with visit status (White-Gray-Black Marking)** to detect cycles.  
The time complexity is **O(V + E)**.

---

#### **Undirected Graph - Union-Find**
```cpp
#include <iostream>
#include <vector>
using namespace std;

class UnionFind {
private:
    vector<int> parent, rank;

public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]); // Path compression
        return parent[x];
    }

    bool unite(int x, int y) {
        int rootX = find(x), rootY = find(y);
        if (rootX == rootY) return false;  // Cycle detected

        if (rank[rootX] > rank[rootY])
            parent[rootY] = rootX;
        else if (rank[rootX] < rank[rootY])
            parent[rootX] = rootY;
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        return true;
    }
};

bool hasCycle(int V, vector<pair<int, int>>& edges) {
    UnionFind uf(V);
    for (auto [u, v] : edges) {
        if (!uf.unite(u, v)) return true;  // Cycle detected
    }
    return false;
}

int main() {
    int V = 4;
    vector<pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};  // Cycle

    cout << (hasCycle(V, edges) ? "Cycle found" : "No cycle") << endl;
    return 0;
}
```
**Output:**
```
Cycle found
```
This approach uses **Union-Find with path compression**.  
The time complexity is **approximately O(α(n))**, where **α(n)** is the inverse Ackermann function, which is nearly constant in practical scenarios.
---

## **8. When to Use Overloading vs Overriding?**
- **Overloading**  
  - **Same function name, different parameters (resolved at compile-time)**  
  - Example: Java’s `print(int x)` vs. `print(String s)`  

- **Overriding**  
  - **Subclass modifies the superclass’s method (resolved at runtime)**  
  - Example: `Animal.speak()` overridden in `Dog` as `bark()`  

---

## **9. Have You Used Encapsulation in a Project? Why?**
- **Encapsulation**:
  1. **Hides internal details**, preventing direct data modification  
  2. **Protects data** (e.g., using `private` fields, getters, and setters)  
  3. **Improves maintainability** (modifications affect only internal class logic)  

- **In a Team Environment**:  
  - Encapsulation ensures **only necessary APIs are exposed**, preventing accidental modifications  
  - **Access control (public/private/protected)** manages data access  

---

