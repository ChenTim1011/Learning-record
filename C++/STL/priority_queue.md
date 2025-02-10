### **How Priority Queue is Implemented in STL?**

In C++ Standard Template Library (STL), the **priority queue** is implemented using a **heap** data structure, most commonly a **binary heap**. The heap is usually stored in a container like `std::vector`. Here's a detailed explanation of how it's implemented, followed by examples.

---

### **1. What is a Priority Queue?**

A **priority queue** is a type of container that allows efficient retrieval of the highest (or lowest) priority element, which is always located at the top of the queue. It supports two main operations:
- **Insert**: Adds an element to the queue.
- **Remove**: Removes the element with the highest priority (usually the top element).

In STL, the priority queue is implemented in such a way that elements are automatically arranged according to their priority, typically using a **max-heap** (for max-priority queue) or a **min-heap** (for min-priority queue).

---

### **2. Data Structure Behind STL Priority Queue**

The STL `std::priority_queue` uses a **heap** to implement the priority queue. A heap is a **binary tree** where the parent node always satisfies a condition with respect to its children:
- In a **max-heap**, the parent node's value is always greater than or equal to the values of its children.
- In a **min-heap**, the parent node's value is always less than or equal to the values of its children.

#### **Heap Representation in `std::priority_queue`**

The heap is typically represented using a dynamic array (like `std::vector`). Elements are arranged so that each parent node has a higher (in max-heap) or lower (in min-heap) priority than its children. This allows efficient access to the top element in constant time \(O(1)\), while insertions and deletions are performed in logarithmic time \(O(\log n)\).

---

### **3. Operations on Priority Queue**

The **priority queue** in STL supports the following operations:

#### **Push Operation (`push()`)**

- When a new element is added to the priority queue, it's inserted at the end of the container (typically the vector).
- Then, the heap property is restored by **heapifying** the queue. This ensures the top element remains the highest (or lowest) priority element.
  
#### **Pop Operation (`pop()`)**

- When an element is removed from the priority queue, the top element is taken out (i.e., the element with the highest priority).
- After removing the top element, the heap property is restored by moving the last element to the top and **heapifying** the queue to maintain the heap structure.

#### **Top Operation (`top()`)**

- Returns the element with the highest priority (the root of the heap). This operation takes constant time \(O(1)\).

#### **Empty Operation (`empty()`)**

- Checks if the priority queue is empty. This is a basic operation with constant time \(O(1)\).

---

### **4. Example Usage of Priority Queue in STL**

Here is an example demonstrating how to use a `std::priority_queue` in C++:

#### **Max-Priority Queue Example (Default)**

By default, `std::priority_queue` implements a **max-heap**. This means the highest element will always be at the top.

```cpp
#include <iostream>
#include <queue>
#include <vector>

int main() {
    // Create a priority queue (max-heap by default)
    std::priority_queue<int> pq;

    // Inserting elements
    pq.push(10);
    pq.push(20);
    pq.push(5);
    pq.push(15);

    // Accessing the top element (max value)
    std::cout << "Top element: " << pq.top() << std::endl;  // Output: 20

    // Removing the top element
    pq.pop();

    std::cout << "New top element: " << pq.top() << std::endl;  // Output: 15

    return 0;
}
```

#### **Min-Priority Queue Example (Custom Comparator)**

You can customize the priority queue to create a **min-heap** by providing a custom comparator.

```cpp
#include <iostream>
#include <queue>
#include <vector>

// Custom comparator for min-heap
struct Compare {
    bool operator()(int a, int b) {
        return a > b;  // Return true if a is greater than b, creating a min-heap
    }
};

int main() {
    // Create a priority queue with min-heap behavior
    std::priority_queue<int, std::vector<int>, Compare> pq;

    // Inserting elements
    pq.push(10);
    pq.push(20);
    pq.push(5);
    pq.push(15);

    // Accessing the top element (min value)
    std::cout << "Top element: " << pq.top() << std::endl;  // Output: 5

    // Removing the top element
    pq.pop();

    std::cout << "New top element: " << pq.top() << std::endl;  // Output: 10

    return 0;
}
```

---

### **5. Time Complexity of Operations**

| **Operation**  | **Time Complexity** |
|----------------|---------------------|
| **push()**     | O(log n)            |
| **pop()**      | O(log n)            |
| **top()**      | O(1)                |
| **empty()**    | O(1)                |

---

### **6. Summary**

- The STL `std::priority_queue` is implemented using a **heap** (usually a binary heap).
- By default, it uses a **max-heap**, but you can easily switch to a **min-heap** by providing a custom comparator.
- The priority queue maintains the heap property by performing **heapify** operations on insertions and removals.
- The priority queue allows constant time access to the top element, with logarithmic time complexity for insertion and removal.

This makes the priority queue an excellent choice for scenarios where you frequently need to access the highest (or lowest) priority element efficiently.