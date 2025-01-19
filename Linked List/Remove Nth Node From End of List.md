[Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)


### Problem Explanation 

You are given a singly linked list and an integer `n`. The task is to remove the N-th node from the end of the list and return the head of the modified list.

For example:
- Input: `head = [1,2,3,4,5], n = 2`
- Output: `head = [1,2,3,5]`

---

### Approach: Two-Pointer Technique

The solution uses the **two-pointer technique** with a **dummy head** to simplify edge cases:
1. Use two pointers: `fast` and `slow`, both initialized to a dummy node pointing to the head.
2. Move the `fast` pointer `n+1` steps forward so that the `fast` pointer and `slow` pointer are exactly `n` nodes apart.
3. Move both pointers one step at a time until `fast` reaches the end of the list. At this point, `slow` points to the node before the one to be removed.
4. Update the `next` pointer of the `slow` node to skip the node that needs to be deleted.
5. Return the updated list starting from the node after the dummy head.

---

### C++ Code with Detailed Comments

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // Step 1: Create a dummy node to simplify edge case handling (e.g., removing the head)
        ListNode* dummyHead = new ListNode(0); // Dummy node with value 0
        dummyHead->next = head;               // Point dummy node to the head of the list
        
        // Step 2: Initialize two pointers: `fast` and `slow`, both pointing to the dummy node
        ListNode* slow = dummyHead;
        ListNode* fast = dummyHead;
        
        // Step 3: Move the `fast` pointer `n+1` steps forward
        for (int i = 0; i <= n; ++i) {
            if (fast != nullptr) {
                fast = fast->next;
            }
        }
        
        // Step 4: Move both pointers forward until `fast` reaches the end
        // At this point, `slow` will point to the node before the one to be removed
        while (fast != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }
        
        // Step 5: Delete the node after the `slow` pointer
        ListNode* nodeToDelete = slow->next;    // Node to be removed
        slow->next = nodeToDelete->next;       // Skip the node to delete
        delete nodeToDelete;                   // Free the memory of the deleted node
        
        // Step 6: Return the modified list, starting from the node after the dummy head
        ListNode* result = dummyHead->next;    // The real head of the modified list
        delete dummyHead;                      // Free the memory of the dummy node
        return result;
    }
};
```

---

### Detailed Explanation of the Code

#### Step-by-Step Execution
1. **Dummy Node:**  
   - A dummy node (`dummyHead`) is created and points to the original head of the list. This simplifies edge cases such as removing the head itself.
   - For example, if the list is `1 -> 2 -> 3 -> 4 -> 5`, the dummy node makes it look like `0 -> 1 -> 2 -> 3 -> 4 -> 5`.

2. **Fast Pointer Movement:**  
   - The `fast` pointer moves `n+1` steps ahead.  
   - Why `n+1`? This ensures that the gap between `fast` and `slow` pointers is `n`, so when `fast` reaches the end, `slow` will be just before the node to delete.
   - For example, with `n=2`, `fast` will move 3 steps forward.

3. **Both Pointers Move Together:**  
   - Both `fast` and `slow` pointers move one step at a time until `fast` reaches the end of the list.
   - At this point, `slow` points to the node immediately before the one to be deleted.

4. **Node Deletion:**  
   - The `next` pointer of the `slow` node is updated to skip the node to be deleted.
   - The memory of the deleted node is freed using `delete`.

5. **Return Result:**  
   - The modified list is returned starting from the node after the dummy head.

---

### Complexity Analysis

#### Time Complexity: \(O(n)\)
- The algorithm requires a single traversal of the list. The `fast` pointer moves \(n+1\) steps, and the second traversal ends when the `fast` pointer reaches the end.

#### Space Complexity: \(O(1)\)
- Only a constant amount of extra space is used for the two pointers and the dummy node.

---

### Example Execution

#### Input:
`head = [1, 2, 3, 4, 5]`, `n = 2`

#### Execution Steps:
1. Add dummy: `0 -> 1 -> 2 -> 3 -> 4 -> 5`
2. Move `fast` 3 steps forward (`n+1` steps):
   - After first step: `fast = 1`
   - After second step: `fast = 2`
   - After third step: `fast = 3`
3. Move both `fast` and `slow` until `fast` reaches the end:
   - After first move: `fast = 4`, `slow = 1`
   - After second move: `fast = 5`, `slow = 2`
   - After third move: `fast = NULL`, `slow = 3`
4. Remove `slow->next` (node `4`):
   - Update `slow->next` to skip `4` and point to `5`.
   - Free the memory of node `4`.
5. Result: `1 -> 2 -> 3 -> 5`

#### Output:
`[1, 2, 3, 5]`

---

### Edge Cases
1. **List has only one node:**
   - Input: `head = [1]`, `n = 1`
   - Output: `[]` (return an empty list).
2. **Removing the head node:**
   - Input: `head = [1, 2, 3]`, `n = 3`
   - Output: `[2, 3]`.
3. **List has two nodes:**
   - Input: `head = [1, 2]`, `n = 1`
   - Output: `[1]`.

