[Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/description/)

## Method 1: Without Dummy Head

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        // Remove all nodes from the beginning of the list that have the target value
        while(head != NULL && head->val == val) {
            ListNode* temp = head;
            head = head->next;
            delete temp; // Free the memory of the removed node
        }

        ListNode* cur = head; // Initialize the current pointer to the head of the list

        // Traverse the list and remove nodes with the target value
        while(cur != NULL && cur->next != NULL) {
            if(cur->next->val == val) {
                ListNode* temp = cur->next;
                cur->next = cur->next->next; // Bypass the node with the target value
                delete temp; // Free the memory of the removed node
            } else {
                cur = cur->next; // Move to the next node
            }
        }

        return head; // Return the modified list
    }
};
```

Explanation:
- This method removes elements from the linked list without using a dummy head node.
- First, it removes all nodes from the beginning of the list that have the target value.
- Then, it traverses the list and removes nodes with the target value by bypassing them and freeing their memory.
- The time complexity is O(n) where n is the number of nodes in the list, and the space complexity is O(1) as no extra space is used.

## Method 2: With Dummy Head

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* dummyhead = new ListNode(0); // Create a dummy head node
        dummyhead->next = head; // Point the dummy head to the actual head of the list

        ListNode* cur = dummyhead; // Initialize the current pointer to the dummy head

        // Traverse the list and remove nodes with the target value
        while(cur != NULL && cur->next != NULL) {
            if(cur->next->val == val) {
                ListNode* temp = cur->next;
                cur->next = cur->next->next; // Bypass the node with the target value
                delete temp; // Free the memory of the removed node
            } else {
                cur = cur->next; // Move to the next node
            }
        }

        head = dummyhead->next; // Update the head to the next node of the dummy head
        delete dummyhead; // Free the memory of the dummy head

        return head; // Return the modified list
    }
};
```

Explanation:
- This method uses a dummy head node to simplify the removal process, especially for nodes at the beginning of the list.
- The dummy head node points to the actual head of the list.
- The current pointer is initialized to the dummy head, and the list is traversed to remove nodes with the target value.
- After traversal, the head is updated to the next node of the dummy head, and the dummy head is deleted to free memory.
- The time complexity is O(n) where n is the number of nodes in the list, and the space complexity is O(1) as no extra space is used.

Comparison:
- Both methods have the same time and space complexity.
- The method with the dummy head is more straightforward and handles edge cases (like removing the head node) more elegantly.
- The method without the dummy head requires additional checks to handle the removal of nodes at the beginning of the list.