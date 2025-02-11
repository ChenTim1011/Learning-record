[Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)

## Iterative Method

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
    ListNode* reverseList(ListNode* head) {
        ListNode* temp; // Temporary pointer to store the next node
        ListNode* pre = NULL; // Pointer to the previous node, initially NULL
        ListNode* cur = head; // Pointer to the current node, initially the head

        // Iterate through the list
        while(cur != nullptr) {
            temp = cur->next; // Store the next node
            cur->next = pre; // Reverse the current node's pointer
            pre = cur; // Move the previous pointer to the current node
            cur = temp; // Move the current pointer to the next node
        }

        return pre; // Return the new head of the reversed list
    }
};
```

Explanation:
- This method reverses a linked list iteratively.
- It uses three pointers: `pre` (previous node), `cur` (current node), and `temp` (temporary pointer to store the next node).
- The `cur` pointer traverses the list, and for each node, its `next` pointer is reversed to point to the `pre` node.
- The `pre` pointer is then moved to the current node, and the `cur` pointer is moved to the next node.
- The process continues until all nodes are reversed.
- The time complexity is O(n) where n is the number of nodes in the list, and the space complexity is O(1) as no extra space is used.

## Recursive Method

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
    // Helper function to reverse the list recursively
    ListNode* reverse(ListNode* pre, ListNode* cur) {
        if(cur == NULL) return pre; // Base case: if current node is NULL, return the previous node
        ListNode* temp = cur->next; // Store the next node
        cur->next = pre; // Reverse the current node's pointer
        return reverse(cur, temp); // Recur with the current node as previous and the next node as current
    }

    ListNode* reverseList(ListNode* head) {
        return reverse(NULL, head); // Call the helper function with NULL as previous and head as current
    }
};
```

Explanation:
- This method reverses a linked list recursively.
- It uses a helper function `reverse` that takes two pointers: `pre` (previous node) and `cur` (current node).
- The base case of the recursion is when the `cur` pointer is NULL, in which case the `pre` pointer is returned as the new head of the reversed list.
- For each recursive call, the `cur` node's `next` pointer is reversed to point to the `pre` node.
- The function then recurs with the `cur` node as the new `pre` and the next node as the new `cur`.
- The time complexity is O(n) where n is the number of nodes in the list, and the space complexity is O(n) due to the recursion stack.

Comparison:
- Both methods have the same time complexity of O(n).
- The iterative method has a space complexity of O(1), while the recursive method has a space complexity of O(n) due to the recursion stack.
- The iterative method is generally preferred for its simplicity and lower space usage.