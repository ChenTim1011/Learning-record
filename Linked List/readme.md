# All About Linked Lists

### What Is a Linked List?

A **linked list** is a linear data structure where elements (called *nodes*) are linked together using pointers. Each node in a linked list contains two parts:  
1. **Data field**: Stores the actual data.  
2. **Pointer field**: Holds a reference (or pointer) to the next node in the sequence.

The last node in a linked list points to `null` (or `nullptr` in C++), indicating the end of the list.

The starting point of a linked list is referred to as the **head node** or simply `head`.

**Illustration**: Linked List (Diagram not included here)

---

### Types of Linked Lists

There are several variations of linked lists:

1. **Singly Linked List**  
   This is the simplest form of a linked list. In a singly linked list, each node contains a pointer to the next node in the sequence. Traversal is one-directional, from the head to the last node.

2. **Doubly Linked List**  
   A doubly linked list extends the concept of a singly linked list by adding another pointer. Each node contains:  
   - A pointer to the **next node**.  
   - A pointer to the **previous node**.  

   This allows bidirectional traversal, meaning you can move both forward and backward through the list.

3. **Circular Linked List**  
   A circular linked list connects the last node back to the first node, forming a circular structure.  
   - In a singly circular linked list, the last node points to the first node.  
   - In a doubly circular linked list, both the first and last nodes are linked to each other.  

   Circular linked lists are often used in problems like the **Josephus problem**.

---

### Memory Representation of Linked Lists

Unlike arrays, which have contiguous memory allocation, the nodes of a linked list are **scattered across different memory locations**. Each node is connected to the next through its pointer field.

- Arrays require contiguous memory, but linked lists do not.  
- The memory location for each node in a linked list depends on the operating system's memory allocation process.

**Illustration**: Memory distribution (Diagram not included here)

Example:  
A linked list might start at memory address 2 and end at memory address 7, with nodes scattered in between. Each node’s pointer field holds the address of the next node in sequence.

---

### Defining a Linked List Node

To implement a linked list, you first need to define what a node is. In many interview questions, the definition of a node is pre-provided. However, if you need to write it yourself, here’s how you can define a node in C++:

**Example: Singly Linked List Node in C++**
```cpp
struct ListNode {
    int val;              // Data field
    ListNode* next;       // Pointer to the next node

    // Constructor to initialize the node
    ListNode(int x) : val(x), next(NULL) {}
};
```

#### With or Without a Constructor?

You can choose to define your own constructor or rely on the default constructor provided by C++.

1. **Using a Custom Constructor**  
   Example:  
   ```cpp
   ListNode* head = new ListNode(5);
   ```
   This initializes the node with a value of `5`.

2. **Using the Default Constructor**  
   Example:  
   ```cpp
   ListNode* head = new ListNode();
   head->val = 5;  // Manually set the value
   ```
   Without a custom constructor, you must manually assign values to the node after its creation.

---

### Common Operations on Linked Lists

1. **Deleting a Node**  
   To delete a node (e.g., node `D`), simply update the `next` pointer of the preceding node (`C`) to skip over `D` and point to the next node (`E`).

   Example:  
   If you delete node `D`, the memory allocated for it is still reserved in most languages. In C++, you should explicitly release this memory using `delete`.  
   In languages like Java or Python, the garbage collector handles memory deallocation automatically.

2. **Adding a Node**  
   To add a new node, adjust the `next` pointer of the relevant nodes to include the new node in the chain.

**Time Complexity**:
- Adding or deleting a node is an **O(1)** operation because no elements need to be shifted, as is the case with arrays.  
- Searching for a node, however, requires traversing the list, which takes **O(n)** time.

---

### Performance Comparison: Linked Lists vs. Arrays

| **Feature**              | **Array**                         | **Linked List**                        |
|--------------------------|------------------------------------|---------------------------------------|
| **Memory Allocation**    | Contiguous                        | Scattered                             |
| **Size**                 | Fixed at definition               | Dynamic, can grow/shrink as needed    |
| **Insertion/Deletion**   | Expensive (`O(n)` due to shifting)| Efficient (`O(1)`)                    |
| **Random Access**        | Possible (`O(1)` via index)       | Not possible (must traverse: `O(n)`)  |

**Key Takeaways**:
- Use arrays when data is static, and frequent random access is needed.  
- Use linked lists when data size is dynamic, and frequent insertions/deletions are required.

---
Reference:
[Linkedlist](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E9%93%BE%E8%A1%A8%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md)
