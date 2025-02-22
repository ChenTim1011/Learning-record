[Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/description/)

### Problem Explanation 
You are given a singly linked list, and the task is to swap every two adjacent nodes and return the new head of the list. Importantly:
- You cannot simply swap the **values** of the nodes.
- You must perform the actual node swapping by manipulating the pointers.

For example:
- Input: `1 -> 2 -> 3 -> 4`
- Output: `2 -> 1 -> 4 -> 3`

---

### Approach
To solve this problem, we simulate the swapping process while using a **dummy head**. The dummy head simplifies handling edge cases, such as when the head node is swapped or the list is empty.

The algorithm involves:
1. Using a dummy node (`dummyHead`) to simplify operations on the head.
2. Iterating through the list in pairs using a pointer (`cur`).
3. Swapping adjacent nodes by reassigning their `next` pointers, ensuring all links are maintained.
4. Moving the pointer forward after each pair swap.

---

### Steps (with a diagram)
Consider a list `1 -> 2 -> 3 -> 4`:
1. Add a dummy node before the head: `0 -> 1 -> 2 -> 3 -> 4`.
2. Use a pointer `cur` initialized to the dummy node.
3. At each step:
   - Identify the two nodes to swap (`cur->next` and `cur->next->next`).
   - Reassign pointers to swap them:
     - Step 1: Point `cur->next` to the second node (`2`).
     - Step 2: Point the second node's `next` to the first node (`1`).
     - Step 3: Connect the first node (`1`) to the rest of the list (`3`).
   - Move `cur` two steps forward.

After the swaps:
- The list becomes: `0 -> 2 -> 1 -> 4 -> 3`.

Finally, return `dummyHead->next` as the new head.

---

### C++ Code with Detailed Comments

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        // Step 1: Create a dummy node pointing to the head of the list
        ListNode* dummyHead = new ListNode(0); // This simplifies edge cases
        dummyHead->next = head; // Point dummy node to the original head
        ListNode* cur = dummyHead; // Initialize the pointer to the dummy node
        
        // Step 2: Traverse the list and swap nodes in pairs
        while (cur->next != nullptr && cur->next->next != nullptr) {
            // Identify the nodes to be swapped
            ListNode* first = cur->next;        // First node of the pair
            ListNode* second = cur->next->next; // Second node of the pair
            ListNode* nextPair = second->next; // The node after the pair
            
            // Step 3: Swap the nodes
            cur->next = second;          // Step 1: Connect cur to the second node
            second->next = first;        // Step 2: Point the second node to the first
            first->next = nextPair;      // Step 3: Connect the first node to the rest of the list
            
            // Move the pointer forward to the next pair
            cur = first; // `first` is now the last node in the swapped pair
        }
        
        // Step 4: Return the new head (the node after the dummy head)
        ListNode* result = dummyHead->next;
        delete dummyHead; // Clean up the dummy node to avoid memory leaks
        return result;
    }
};
```

---

### Explanation of the Code

#### Key Variables:
- **`dummyHead`:** A dummy node added before the head to simplify pointer manipulation.
- **`cur`:** Tracks the node before the current pair being swapped.
- **`first` and `second`:** The two nodes being swapped in the current iteration.
- **`nextPair`:** The node following the current pair, used to reconnect the list after the swap.

#### Key Steps:
1. **Dummy Node Initialization:**
   - The dummy node simplifies the process by ensuring we don't need to treat the head node as a special case.
2. **Swapping:**
   - Update `cur->next` to point to the second node.
   - Make the second node point to the first.
   - Connect the first node to the node following the current pair.
3. **Pointer Movement:**
   - Move the `cur` pointer two steps forward to prepare for the next pair swap.
4. **Edge Cases:**
   - If the list has fewer than two nodes, the loop is skipped, and the original head is returned.

---

### Complexity Analysis
- **Time Complexity:** \(O(n)\), where \(n\) is the number of nodes in the list. Each node is visited once during the traversal.
- **Space Complexity:** \(O(1)\), as no additional data structures are used; only a few pointers are manipulated.

---

### Edge Cases
1. **Empty List:** If the list is empty (`head == nullptr`), the dummy head is created but the function returns immediately.
2. **Single Node:** If the list has one node, the loop is skipped, and the original head is returned.
3. **Odd Number of Nodes:** The last node remains unchanged, as swapping requires a pair.

---

### Example Execution
#### Input:
`1 -> 2 -> 3 -> 4`

#### Execution:
1. After the first swap: `2 -> 1 -> 3 -> 4`
2. After the second swap: `2 -> 1 -> 4 -> 3`

#### Output:
`2 -> 1 -> 4 -> 3`

This approach ensures clarity, simplicity, and optimal performance.