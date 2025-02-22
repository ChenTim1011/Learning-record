[Recover a Tree From Preorder Traversal](https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/description/)

# **Problem Explanation**
The problem gives us a **preorder traversal** of a binary tree in the form of a string. The string follows these rules:

1. **Each node is represented by an integer** (`1 <= Node.val <= 10^9`).
2. **Each node is prefixed by `D` dashes (`-`), where `D` is the depth of the node**:
   - The **root node** has depth `0`, so it has **no dashes**.
   - A node with **depth `D` has `D` dashes** before it.
3. **If a node has only one child, it is always the left child**.

---

## **Example 1**
### **Input:**  
```plaintext
"1-2--3--4-5--6--7"
```
### **Expected Output (Tree Representation):**
```
       1
      / \
     2   5
    / \  / \
   3   4 6  7
```
### **Preorder Traversal Output:**
```plaintext
[1, 2, 5, 3, 4, 6, 7]
```
---

## **Example 2**
### **Input:**  
```plaintext
"1-2--3---4-5--6---7"
```
### **Expected Output (Tree Representation):**
```
       1
      / \
     2   5
    /   / 
   3   6   
  /     \
 4       7
```
### **Preorder Traversal Output:**
```plaintext
[1, 2, 5, 3, null, 6, null, 4, null, 7]
```
---

## **Approach**
The idea is to:
1. **Parse the string** to extract integers and their depths.
2. **Use recursion (DFS) to reconstruct the binary tree**.
3. **Use a global index `i`** to traverse through the string.

### **Recursive Approach**
- We **start from index `i = 0`**.
- **Count the number of `-` characters** to determine the depth.
- **Extract the integer value** of the node.
- **Recursively build the left and right child nodes** (depth increases by `1` for each level).
- **If the depth is less than expected, backtrack** (reset `i` and return `nullptr`).

---

## **Implementation (C++)**
```cpp
class Solution {
public:
    int i = 0; // Global index for traversing the string

    TreeNode* recoverFromPreorder(string& T, int depth = 0) {
        if (i >= T.size()) return nullptr; // Base case: end of string

        int D = 0; // Count dashes to determine depth
        while (i < T.size() && T[i] == '-') {
            D++, i++;
        }

        // If the current depth is less than expected, backtrack
        if (D < depth) {
            i -= D; // Reset i (go back)
            return nullptr;
        }

        // Read the node value
        int val = 0;
        while (i < T.size() && isdigit(T[i])) {
            val = val * 10 + (T[i] - '0');
            i++;
        }

        // Create the node
        TreeNode* node = new TreeNode(val);

        // Recursively recover left and right children
        node->left = recoverFromPreorder(T, depth + 1);
        node->right = recoverFromPreorder(T, depth + 1);

        return node;
    }
};
```

---

## **Code Walkthrough**
### **Step 1: Counting Dashes**
```cpp
int D = 0; // Count dashes to determine depth
while (i < T.size() && T[i] == '-') {
    D++, i++;
}
```
- We count the number of `-` characters to determine **the depth of the current node**.

---

### **Step 2: Backtracking if Depth is Less Than Expected**
```cpp
if (D < depth) {
    i -= D; // Reset i
    return nullptr;
}
```
- If the **current depth is smaller than expected**, it means **we have moved too deep into the tree**.
- We **reset `i` back** and return `nullptr`.

---

### **Step 3: Extracting Node Value**
```cpp
int val = 0;
while (i < T.size() && isdigit(T[i])) {
    val = val * 10 + (T[i] - '0');
    i++;
}
```
- Read the **integer value of the node** from the string.
- `val = val * 10 + (T[i] - '0')` ensures that we correctly handle multi-digit numbers.

---

### **Step 4: Recursive Calls for Left and Right Subtrees**
```cpp
node->left = recoverFromPreorder(T, depth + 1);
node->right = recoverFromPreorder(T, depth + 1);
```
- Since **preorder traversal processes the root first**, we:
  1. Recursively construct the **left subtree** first.
  2. Then, recursively construct the **right subtree**.

---

## **Time and Space Complexity**
### **Time Complexity:**
- **`O(n)`**, since each character in the input string is processed exactly once.
- Each node is **created once and visited once**, making the algorithm linear.

### **Space Complexity:**
- **`O(n)`**, due to **recursion stack space** in the worst case (skewed tree).
- The worst-case scenario is when the tree degenerates into a linked list.

---

## **Example Test Cases**
```cpp
void printTree(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";
    printTree(root->left);
    printTree(root->right);
}

int main() {
    Solution sol;
    string s1 = "1-2--3--4-5--6--7";
    TreeNode* root1 = sol.recoverFromPreorder(s1);
    printTree(root1); // Expected Output: 1 2 3 4 5 6 7

    string s2 = "1-2--3---4-5--6---7";
    TreeNode* root2 = sol.recoverFromPreorder(s2);
    printTree(root2); // Expected Output: 1 2 3 4 5 6 7

    string s3 = "1-401--349---90--88";
    TreeNode* root3 = sol.recoverFromPreorder(s3);
    printTree(root3); // Expected Output: 1 401 349 90 88
}
```

---

## **Summary**
1. **Count the dashes** (`-`) to determine the **depth** of the node.
2. **Extract the node value** (integer).
3. **Backtrack if needed** (if depth is too shallow).
4. **Recursively construct the left and right subtrees**.
5. **Use a global index `i`** to keep track of the current position in the string.

This **recursive approach** efficiently reconstructs the tree while **ensuring correct node placement** based on preorder traversal.

