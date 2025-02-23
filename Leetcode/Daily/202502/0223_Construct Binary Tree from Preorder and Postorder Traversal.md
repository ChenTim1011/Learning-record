[Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/description/)

## **Understanding the Problem**  
We are given two arrays:  
- **Preorder traversal** (`preorder`): Visits nodes in **root → left → right** order.  
- **Postorder traversal** (`postorder`): Visits nodes in **left → right → root** order.  

Our goal is to reconstruct the **original binary tree** from these two traversals.

### **Key Observations**  
1. The **first element** of `preorder` is always the **root** of the tree.  
2. The **last element** of `postorder` is also the **root** of the tree.  
3. The **left subtree** is formed from the next node in `preorder` (i.e., `preorder[1]`), which must be the left child of the root.  
4. Using `postorder`, we can determine where the left subtree ends by finding `preorder[1]` inside `postorder`.  
5. The remaining nodes in `preorder` and `postorder` form the **right subtree**.  

---

## **Approach (Iterative Using Stack)**
We will use a **stack** to construct the tree dynamically, processing nodes from `preorder` and checking their completion with `postorder`.

### **Steps**
1. Create a **stack** and push the **root node** (first element of `preorder`).
2. Traverse `preorder`, inserting nodes as the **left** or **right child** of the **top node in the stack**.
3. If the **top node in the stack** matches the current `postorder` element, **pop it** because its subtree is complete.
4. Continue this process until all nodes are processed.

---

## **Time & Space Complexity**
- **Time Complexity:** \(O(n)\), because each node is pushed and popped from the stack **at most once**.  
- **Space Complexity:** \(O(n)\), as the stack stores **at most \(O(n)\) nodes**.

---

## **Code Implementation**
```cpp
class Solution {
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
        stack<TreeNode*> nodes;
        TreeNode* root = new TreeNode(preorder[0]); // Create root
        nodes.push(root);

        int preIndex = 1, postIndex = 0;

        while (!nodes.empty()) {
            TreeNode* current = nodes.top();

            if (current->val == postorder[postIndex]) { 
                nodes.pop(); // If subtree is complete, pop it
                postIndex++;
            } else {
                TreeNode* newNode = new TreeNode(preorder[preIndex++]);
                if (!current->left) {
                    current->left = newNode; // Assign left child first
                } else {
                    current->right = newNode; // Assign right child next
                }
                nodes.push(newNode);
            }
        }
        return root;
    }
};
```

---

## **Example Walkthrough**
### **Example 1**  
**Input:**  
```cpp
preorder = [1,2,4,5,3,6,7] 
postorder = [4,5,2,6,7,3,1]
```
#### **Step-by-step Execution**
1. **Create root** (`1`) → Push to stack  
2. Process **preorder[1]** → `2` → Becomes `1`’s **left child**  
3. Process **preorder[2]** → `4` → Becomes `2`’s **left child**  
4. Since `4 == postorder[0]`, pop `4` (subtree complete)  
5. Process **preorder[3]** → `5` → Becomes `2`’s **right child**  
6. Since `5 == postorder[1]`, pop `5` (subtree complete)  
7. Since `2 == postorder[2]`, pop `2` (subtree complete)  
8. Process **preorder[4]** → `3` → Becomes `1`’s **right child**  
9. Process **preorder[5]** → `6` → Becomes `3`’s **left child**  
10. Since `6 == postorder[3]`, pop `6` (subtree complete)  
11. Process **preorder[6]** → `7` → Becomes `3`’s **right child**  
12. Since `7 == postorder[4]`, pop `7`  
13. Since `3 == postorder[5]`, pop `3`  
14. Since `1 == postorder[6]`, pop `1` (tree completed)  

**Final Output:**
```
       1
      / \
     2   3
    / \  / \
   4  5 6  7
```

---

## **Summary**
- We **use `preorder` to construct nodes** in root-left-right order.  
- We **use `postorder` to track subtree completion** by checking `stack.top() == postorder[postIndex]`.  
- **Stack-based simulation** ensures efficient tree reconstruction.