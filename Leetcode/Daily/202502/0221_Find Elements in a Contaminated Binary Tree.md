[Find Elements in a Contaminated Binary Tree](https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/description/)

## **Problem Explanation**
### **1. Understanding the Contaminated Tree**
We are given a binary tree where **all node values have been contaminated and set to `-1`**. However, we can recover the original values using these rules:
- The root node **always** has the value `0`.
- If a node has the value `x`:
  - Its **left child** has the value `2 * x + 1`
  - Its **right child** has the value `2 * x + 2`

### **2. Operations Required**
We need to implement a class `FindElements` with two main functionalities:
1. **Constructor: `FindElements(TreeNode* root)`**
   - Initializes the object by **recovering the tree** from the contaminated state.
   - Uses a **hash map** (C++) or **hash set** (Python) to store recovered values for quick lookup.

2. **Find function: `bool find(int target)`**
   - Checks if `target` exists in the recovered tree.
   - Uses a **hash map lookup** to ensure **O(1) search time**.

---

## **Example Walkthrough**
### **Example 1**
#### **Input**
```cpp
["FindElements", "find", "find"]
[[[-1,null,-1]], [1], [2]]
```
#### **Recovered Tree**
1. Root node → `0`
2. Right child → `2 = 2 * 0 + 2`
3. The final tree:
```
    0
     \
      2
```
#### **Find Operations**
- `find(1) → false` (Not in the tree)
- `find(2) → true` (Exists in the tree)

#### **Output**
```cpp
[null, false, true]
```

---

### **Example 2**
#### **Input**
```cpp
["FindElements", "find", "find", "find"]
[[[-1,-1,-1,-1,-1]], [1], [3], [5]]
```
#### **Recovered Tree**
```
        0
       / \
      1   2
     / \
    3   4
```
#### **Find Operations**
- `find(1) → true`
- `find(3) → true`
- `find(5) → false` (Not in the tree)

#### **Output**
```cpp
[null, true, true, false]
```

---

## **Approach: DFS + Hash Map**
### **Steps**
1. **Recover the Tree Using DFS**
   - Assign `root->val = 0`
   - Recursively compute left and right child values:
     - `left = 2 * parent + 1`
     - `right = 2 * parent + 2`
   - Store all values in a **hash map** for quick lookup.

2. **Fast O(1) Search Using Hash Map**
   - `find(target)` simply checks if `target` exists in the hash map.

---

## **C++ Solution**
```cpp
class FindElements {
public:
    unordered_map<int, bool> mp;

    // Recursive function to recover the tree using DFS
    void recover(TreeNode* root) {
        if (!root) return;
        if (root->left) {
            root->left->val = root->val * 2 + 1;
            mp[root->left->val] = true;
            recover(root->left);
        }
        if (root->right) {
            root->right->val = root->val * 2 + 2;
            mp[root->right->val] = true;
            recover(root->right);
        }
    }

    // Constructor: Recovers the tree
    FindElements(TreeNode* root) {
        root->val = 0;
        mp[0] = true;
        recover(root);
    }
    
    // Check if target exists
    bool find(int target) {
        return mp.find(target) != mp.end();
    }
};
```

---

## **Time and Space Complexity**
| Operation | Time Complexity | Space Complexity |
|-----------|---------------|----------------|
| **Recovering the tree (DFS)** | O(N) | O(N) |
| **Finding an element** | O(1) | O(N) |

- **N** is the number of nodes in the tree.
- **Recovering the tree** requires visiting all `N` nodes → O(N).
- **Finding a target** is O(1) due to hash map lookup.
- **Space complexity** is O(N) for storing values.

---

## **Summary**
✅ **DFS to restore the tree**, computing node values recursively  
✅ **Hash Map / Hash Set to store values**, enabling O(1) lookup  
✅ **Time Complexity: O(N) recovery, O(1) find**—efficient even for large inputs  

