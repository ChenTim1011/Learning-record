### The Basics of Stacks and Queues

The principles of stacks and queues are likely familiar to everyone:  
- **Queue**: Follows a **First In, First Out (FIFO)** principle, where elements are processed in the order they arrive.  
- **Stack**: Follows a **Last In, First Out (LIFO)** principle, where the last element added is the first to be removed.  

For example, as shown in the diagrams (not included here), a queue processes elements from the front, while a stack processes elements from the top.

---

### Four Key Questions about Stacks in C++  
Let’s start with four important questions about stacks in C++. These questions encourage you to delve deeper into the stack's implementation in your programming language of choice:  

1. **Is `stack` a container in C++?**  
2. **Which version of the Standard Template Library (STL) does `stack` belong to?**  
3. **How is the `stack` implemented in the STL?**  
4. **Does `stack` provide an iterator to traverse its elements?**  

---

### Common Misunderstandings  
Some developers may only know that stacks and queues are data structures and might be familiar with their basic usage. However, they might not understand the underlying implementation or how these structures relate to the STL.  

Let’s clear these doubts by exploring the fundamental concepts and implementation details.

---

### **Stacks and Queues in the STL**

- **STL Overview**:  
  Stacks and queues are part of the **C++ Standard Template Library (STL)**. However, it’s important to understand which version of the STL you're working with to grasp the details of the implementation.  

- **Three Major STL Versions**:
  1. **HP STL**: The original STL implementation, which is open-source and forms the blueprint for most other versions.  
  2. **P.J. Plauger STL**: An STL version implemented by P.J. Plauger, adopted by Microsoft Visual C++ (non-open-source).  
  3. **SGI STL**: An open-source implementation by Silicon Graphics, widely used in GCC (GNU Compiler Collection). This version is well-documented, making it an excellent resource for understanding STL internals.  

The discussion here focuses on the **SGI STL** implementation.

---

### **Stack Details**  

#### What is a Stack?  
A stack is a LIFO (Last In, First Out) data structure. Its interface provides operations such as `push` (to add an element) and `pop` (to remove an element).  

#### Key Characteristics of Stacks:  
- A stack enforces the LIFO rule, so it does **not** allow traversal of all elements.  
- **Iterators** are not provided for stacks (unlike other containers like `set` or `map`).

#### Why?  
The stack is not a standalone container but rather a **container adapter**. This means that it wraps around an underlying container to provide a specific interface (LIFO behavior) without exposing the underlying structure directly.

---

### **Underlying Implementation of Stacks**  

In the STL, the stack’s implementation relies on another container. The **default underlying container** for a stack is the `deque` (double-ended queue). However, you can replace it with other containers like `vector` or `list`.  

#### Example Implementation with Different Containers:  
```cpp
std::stack<int, std::vector<int>> myStack;  // Stack using vector as the underlying container
```

#### Why `deque` as the Default?  
- A `deque` allows fast insertion and deletion at both ends, making it ideal for stack implementation when only one end is used.  
- The other containers (`vector` or `list`) also work but may have trade-offs in performance based on the use case.

---

### **Queue Details**

#### What is a Queue?  
A queue is a FIFO (First In, First Out) data structure. Its interface provides operations such as `enqueue` (to add an element) and `dequeue` (to remove an element).

#### Key Characteristics of Queues:  
- A queue enforces the FIFO rule, so like a stack, it does **not** allow element traversal.  
- **Iterators** are not provided for queues.  

#### Queue’s Underlying Implementation:  
Similar to the stack, the queue is implemented as a **container adapter** in the STL. Its default underlying container is also the `deque`.  

#### Example with Other Containers:  
```cpp
std::queue<int, std::list<int>> myQueue;  // Queue using list as the underlying container
```

---

### **Understanding Container Adapters**  

Both `stack` and `queue` are **container adapters**, not standalone containers. A **container adapter** provides a specific interface by wrapping around an underlying container. This allows the same logical structure (e.g., stack or queue) to be implemented using different containers (`vector`, `list`, `deque`), depending on performance requirements.  

---

### **Final Thoughts**  

For developers using other programming languages, it’s equally important to explore the underlying implementation of stacks and queues. By understanding how these structures are built, you can deepen your knowledge of data structures and make informed decisions when optimizing your code.  

Refenrence:
[Stack and queue](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E6%A0%88%E4%B8%8E%E9%98%9F%E5%88%97%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md)