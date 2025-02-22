[Design Linked List](https://leetcode.com/problems/design-linked-list/description/)

```c++
class MyLinkedList {
public:
    // Definition of the ListNode structure
    struct ListNode {
        int val; // Value of the node
        ListNode* next; // Pointer to the next node
        ListNode(int x) : val(x), next(nullptr) {} // Constructor to initialize the node
    };

    // Constructor to initialize the linked list
    MyLinkedList() {
        dummy_head = new ListNode(0); // Create a dummy head node
        size = 0; // Initialize the size of the linked list
    }
    
    // Function to get the value of the node at the given index
    int get(int index) {
        if (index < 0 || index >= size) {
            return -1; // Return -1 if the index is invalid
        }
        ListNode* cur = dummy_head->next; // Start from the first actual node
        while (index--) {
            cur = cur->next; // Move to the next node
        }
        return cur->val; // Return the value of the node at the given index
    }
    
    // Function to add a node with the given value at the head of the linked list
    void addAtHead(int val) {
        ListNode* newnode = new ListNode(val); // Create a new node
        newnode->next = dummy_head->next; // Point the new node to the current head
        dummy_head->next = newnode; // Update the dummy head to point to the new node
        size++; // Increment the size of the linked list
    }
    
    // Function to add a node with the given value at the tail of the linked list
    void addAtTail(int val) {
        ListNode* newnode = new ListNode(val); // Create a new node
        ListNode* cur = dummy_head; // Start from the dummy head
        while (cur->next != nullptr) {
            cur = cur->next; // Move to the next node
        }
        cur->next = newnode; // Point the last node to the new node
        size++; // Increment the size of the linked list
    }
    
    // Function to add a node with the given value at the specified index
    void addAtIndex(int index, int val) {
        if (index > size) {
            return; // Do nothing if the index is greater than the size
        }
        if (index < 0) {
            index = 0; // Adjust the index if it is negative
        }
        ListNode* newnode = new ListNode(val); // Create a new node
        ListNode* cur = dummy_head; // Start from the dummy head
        while (index--) {
            cur = cur->next; // Move to the next node
        }
        newnode->next = cur->next; // Point the new node to the next node
        cur->next = newnode; // Point the current node to the new node
        size++; // Increment the size of the linked list
    }
    
    // Function to delete the node at the specified index
    void deleteAtIndex(int index) {
        if (index < 0 || index >= size) {
            return; // Do nothing if the index is invalid!
        }
        ListNode* cur = dummy_head; // Start from the dummy head
        while (index--) {
            cur = cur->next; // Move to the next node
        }
        ListNode* temp = cur->next; // Store the node to be deleted !
        cur->next = cur->next->next; // Bypass the node to be deleted
        delete temp; // Free the memory of the deleted node!
        size--; // Decrement the size of the linked list!
    }

// If private is placed before, dummy_head and size are defined as private members, and the compiler has not yet seen the definition of ListNode, which will cause parsing problems.
// When private is placed after, the definition of struct ListNode is already completed, so ListNode can be directly used in the class constructor or other methods without any issues.

private:
    int size; // Size of the linked list
    ListNode* dummy_head; // Dummy head node to simplify operations
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
```

Explanation:
- The `MyLinkedList` class implements a singly linked list with a dummy head node to simplify operations.
- The `ListNode` structure defines the nodes of the linked list, each containing a value and a pointer to the next node.
- The constructor initializes the linked list with a dummy head node and sets the size to 0.
- The `get` function returns the value of the node at the specified index, or -1 if the index is invalid.
- The `addAtHead` function adds a new node with the given value at the head of the linked list.
- The `addAtTail` function adds a new node with the given value at the tail of the linked list.
- The `addAtIndex` function adds a new node with the given value at the specified index. If the index is greater than the size, the function does nothing. If the index is negative, it is adjusted to 0.
- The `deleteAtIndex` function deletes the node at the specified index. If the index is invalid, the function does nothing.
- The `size` variable keeps track of the number of nodes in the linked list.
- The `dummy_head` node simplifies operations by providing a consistent starting point for traversals and modifications.