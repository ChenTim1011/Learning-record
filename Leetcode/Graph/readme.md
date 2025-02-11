## **What Is a Graph?**
In mathematics and computer science, a **graph** is a structure made up of:
1. **Nodes (or vertices):** Represent points in the graph.
2. **Edges:** Represent connections (or relationships) between pairs of nodes.

### **Basic Concept of a Graph**
- Imagine a **2D plane** where points can be connected by lines. These connections form a graph.
- A graph can consist of just one node or even no nodes at all (called an **empty graph**).

---

## **Types of Graphs**
Graphs are broadly categorized into two main types:

### **1. Undirected Graph**
- Edges in the graph **don’t have a direction**.  
- Example: If Node A is connected to Node B, it means you can travel **both ways** between A and B.  

Illustration:  
```
A --- B --- C  
```  
Here, A and B are connected, and so are B and C.

### **2. Directed Graph (Digraph)**
- Edges in the graph **have a direction**.  
- Example: If there’s a directed edge from Node A to Node B, it means you can travel **from A to B**, but not necessarily from B to A.  

Illustration:  
```
A → B → C  
```
Here, you can only move in the direction of the arrows.

### **Weighted Graphs**
Graphs can also have **weights** assigned to their edges, which represent a cost, distance, or value associated with the connection.  

- **Weighted Undirected Graph:** The weight applies to the edge in both directions.  
- **Weighted Directed Graph:** The weight applies only in the direction of the edge.  

---

## **Degrees of a Node**
The **degree** of a node refers to how many edges are connected to it.

### **In an Undirected Graph:**
- The degree of a node is simply the number of edges connected to it.

Example:  
If Node A has edges connecting it to Nodes B, C, and D, then its degree is 3.

### **In a Directed Graph:**
- Nodes have two degrees:
  1. **In-degree:** Number of edges pointing **into** the node.
  2. **Out-degree:** Number of edges pointing **out** of the node.

Example:  
- If Node A has two edges pointing to it and three edges pointing away from it:
  - **In-degree = 2**
  - **Out-degree = 3**

---

## **Connectivity in Graphs**
Connectivity determines how nodes are linked within a graph.  

### **1. Connected Graph (Undirected)**
- A graph is **connected** if there’s a path between any two nodes.  
- Example: In the graph below, every node is reachable from any other node:  
```
A --- B --- C  
 \         /  
  D ------ E  
```

### **2. Disconnected Graph (Undirected)**
- A graph is **disconnected** if at least one node is **not reachable** from other nodes.  
- Example:  
```
A --- B     C --- D  
```
Here, Nodes A and B are disconnected from Nodes C and D.

### **3. Strongly Connected Graph (Directed)**
- A **strongly connected graph** is a directed graph in which **every node is reachable from every other node** in both directions.
- Example:  
```
A <-> B <-> C  
```
This graph is strongly connected because you can travel between any pair of nodes in both directions.

### **Key Distinction:**
- In an **undirected graph**, being connected is simpler because there are no directions to consider.
- In a **directed graph**, being strongly connected is stricter because it requires paths in **both directions**.

---

## **Connected Components**
A **connected component** is a **maximal connected subgraph** in an undirected graph.  

### What does that mean?  
- It’s a group of nodes such that every node in the group is reachable from every other node in the group.  
- It must be **maximal**: If you add any more nodes, the group will no longer be connected.

Example:  
In the graph below:  
```
1 --- 2 --- 5    3 --- 4 --- 6  
```
- **Connected Component 1:** {1, 2, 5}  
- **Connected Component 2:** {3, 4, 6}  

#### Why {3, 4} is NOT a connected component:  
- {3, 4} is connected, but it’s not **maximal**. Node 6 can also be added to this group without breaking connectivity.  

---

## **Strongly Connected Components**
A **strongly connected component** is a **maximal strongly connected subgraph** in a directed graph.

Example:  
In the graph below:  
```
1 <-> 2 <-> 3   
4 <-> 5 <-> 6 <-> 4  
```
- **Strongly Connected Component 1:** {1, 2, 3}  
- **Strongly Connected Component 2:** {4, 5, 6}  

---

## **Graph Representation**
We can represent graphs using two main approaches:

### **1. Adjacency Matrix**
- Uses a 2D array to represent connections.  
- If there are `n` nodes, the matrix is of size `n x n`.  
  - Entry `[i][j]` is the weight of the edge from Node `i` to Node `j`.  
  - If there’s no edge, the entry is 0 (or some other marker).  

Example:  
For a graph with edges `1 → 2`, `1 → 3`, and `2 → 3`:  
```
   1  2  3  
1  0  1  1  
2  0  0  1  
3  0  0  0  
```

#### Advantages:
- Simple to understand.  
- Fast to check if two nodes are connected.

#### Disadvantages:
- Inefficient for sparse graphs (many nodes, few edges).  
- Requires `O(n^2)` space and time to traverse.

---

### **2. Adjacency List**
- Uses an array of lists.  
  - Each node stores a list of its neighbors.  

Example:  
For the same graph as above:  
```
1 → [2, 3]  
2 → [3]  
3 → []  
```

#### Advantages:
- Space-efficient for sparse graphs.  
- Faster to traverse edges for a given node.

#### Disadvantages:
- Slower to check if two nodes are directly connected.

---

## **Graph Traversal**
There are two main algorithms for graph traversal:

### **1. Depth-First Search (DFS):**
- Explores as far as possible along one path before backtracking.  
- Often implemented with recursion or a stack.  

### **2. Breadth-First Search (BFS):**
- Explores all neighbors of a node before moving to the next level.  
- Often implemented with a queue.  

---

## **Next Steps**
This is a foundational overview of graph theory. In practice, you’ll encounter:
- Algorithms like **Dijkstra's** for shortest paths, **Kruskal's** for minimum spanning trees, etc.
- Real-world problems like **network routing**, **social network analysis**, and **image segmentation**.

Would you like to dive deeper into any specific part, such as traversal algorithms or graph applications?