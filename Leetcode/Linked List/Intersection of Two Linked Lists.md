[Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/description/)

### Problem Explanation
This problem asks us to find the intersection node of two singly linked lists. If the two lists do not intersect, return `null`. It is important to note that the intersection is determined based on the memory address of the nodes, not their values.

### Solution Approach
We can solve this problem using the following steps:

---

#### 1. **Calculate the Lengths of Both Lists**
   - Traverse `listA` and `listB` to calculate their lengths, `lenA` and `lenB`.
   - This will help determine which list is longer and calculate the difference in their lengths (`gap = abs(lenA - lenB)`).

---

#### 2. **Align the Starting Points of Both Lists**
   - Assume `listA` is longer than `listB` (if not, swap them). Move the pointer of the longer list (`curA`) forward by `gap` steps. This ensures that both pointers start from the same relative position (aligned with the end of the lists).
   - Example:
     - If `listA` has length 5 and `listB` has length 3, move `curA` forward by 2 steps.

---

#### 3. **Traverse Both Lists Simultaneously**
   - Move both pointers `curA` and `curB` one step at a time.
   - If the two pointers point to the same node (same memory address), return that node as the intersection point.
   - If both pointers reach the end of their respective lists without meeting, return `null`.

---

#### Key Observation
If two linked lists intersect, their tail sections must be identical. Aligning the pointers and traversing simultaneously ensures that the pointers will meet at the intersection point if it exists.

---

### C++ Solution (With Detailed Comments)

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        // Step 1: Calculate the length of list A
        ListNode* curA = headA;
        ListNode* curB = headB;
        int lenA = 0, lenB = 0;

        // Traverse list A to find its length
        while (curA != nullptr) {
            lenA++;
            curA = curA->next;
        }

        // Traverse list B to find its length
        while (curB != nullptr) {
            lenB++;
            curB = curB->next;
        }

        // Reset the pointers to the heads of the lists
        curA = headA;
        curB = headB;

        // Step 2: Align the starting points
        // Ensure curA points to the longer list
        if (lenB > lenA) {
            swap(lenA, lenB);
            swap(curA, curB);
        }

        // Move curA forward by the length difference
        int gap = lenA - lenB;
        while (gap--) {
            curA = curA->next;
        }

        // Step 3: Traverse both lists together
        while (curA != nullptr && curB != nullptr) {
            if (curA == curB) {
                return curA; // Found the intersection point
            }
            curA = curA->next;
            curB = curB->next;
        }

        return nullptr; // No intersection
    }
};
```

---

### Step-by-Step Explanation

#### Step 1: Calculate Lengths
For example:
- `listA = [4,1,8,4,5]`
- `listB = [5,6,1,8,4,5]`

After traversing both lists:
- `lenA = 5`
- `lenB = 6`

---

#### Step 2: Align the Starting Points
Since `lenB > lenA`, swap the pointers and lengths. Move `curB` forward by `lenB - lenA = 1` step:
- After this, `curB` points to node `6`, and `curA` points to node `4`.

---

#### Step 3: Simultaneous Traversal
Traverse both lists:
- First iteration: `curA = 1`, `curB = 1` (no intersection yet).
- Second iteration: `curA = 8`, `curB = 8` (intersection found).
- Return the node with value `8`.

---

### Time and Space Complexity

#### Time Complexity
- Calculating the lengths of the two lists requires \(O(m + n)\), where \(m\) and \(n\) are the lengths of the two lists.
- Aligning the pointers and traversing the lists requires \(O(\max(m, n))\).
- Total time complexity: **\(O(m + n)\)**.

#### Space Complexity
- We only use a few pointers, so the space complexity is **\(O(1)\)**.

---

### Edge Cases
1. **No Intersection**  
   Example:  
   - `listA = [2,6,4]`
   - `listB = [1,5]`  
   Output: `null`.

2. **Lists Completely Overlap**  
   Example:  
   - `listA = [1,2,3]`
   - `listB = [1,2,3]`  
   Output: Node with value `1`.

3. **One List is Empty**  
   If either `headA == nullptr` or `headB == nullptr`, return `null`.

---

### Summary
This solution uses the two-pointer technique combined with length alignment to efficiently find the intersection point in \(O(m + n)\) time. It avoids unnecessary memory usage by working in constant space and adheres to the requirement of not modifying the original structure of the linked lists.