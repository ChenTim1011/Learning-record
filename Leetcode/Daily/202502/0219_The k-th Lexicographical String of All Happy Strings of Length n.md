[The k-th Lexicographical String of All Happy Strings of Length n](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/description/)

# **Understanding the Problem**  

### **What is a Happy String?**  
A **happy string** is a string that satisfies these conditions:  
1. It only contains **three characters**: `'a'`, `'b'`, and `'c'`.  
2. No **two consecutive** characters are the same.  
   - ✅ `"ab"` (valid)  
   - ❌ `"aa"` (invalid)  
   - ✅ `"aba"` (valid)  
   - ❌ `"abb"` (invalid)  

### **What Do We Need to Do?**  
Given two integers, `n` and `k`:  
1. Generate **all happy strings** of length `n`.  
2. Sort them **in lexicographical order** (dictionary order).  
3. Return the `k`-th string in this order.  
4. If `k` is **greater than the number of possible happy strings**, return an **empty string** (`""`).  

---

# **Example Walkthrough**  

### **Example 1**  
```cpp
Input: n = 1, k = 3
Output: "c"
```
#### **Step 1: Generate Happy Strings of Length `n = 1`**
Since `n = 1`, the only happy strings are:  
```
["a", "b", "c"]
```
#### **Step 2: Find the 3rd String (`k = 3`)**
- The 3rd string is `"c"`, so the output is:  
```cpp
Output: "c"
```

---

### **Example 2**  
```cpp
Input: n = 1, k = 4
Output: ""
```
#### **Step 1: Generate Happy Strings of Length `n = 1`**
Again, the only happy strings are:  
```
["a", "b", "c"]
```
#### **Step 2: Find the 4th String (`k = 4`)**
- There are **only 3 strings**, but `k = 4`, which is **out of range**.  
- So, we return an **empty string** (`""`).

---

### **Example 3**  
```cpp
Input: n = 3, k = 9
Output: "cab"
```
#### **Step 1: Generate Happy Strings of Length `n = 3`**
We generate all happy strings of length `n = 3` **in lexicographical order**:  
```
["aba", "abc", "aca", "acb", "bab", "bac", "bca", "bcb", "cab", "cac", "cba", "cbc"]
```
#### **Step 2: Find the 9th String (`k = 9`)**
- The 9th string in the list is `"cab"`.  
```cpp
Output: "cab"
```

---

# **Solution Approach**
To solve this problem, we can use **backtracking** to generate all valid happy strings.  

## **Steps**
1. **Generate all happy strings of length `n` using recursion.**  
   - Use a helper function that builds strings character by character.  
   - Ensure that no **two consecutive characters are the same**.  
2. **Sort them in lexicographical order.**  
3. **Return the `k`-th string** (if it exists).  

---

### **Code Explanation**
```cpp
class Solution {
private:
    void solve(vector<string> &v, int n, string st, char prevChar) {
        if (st.size() == n) {  // Base case: When we reach length `n`
            v.push_back(st);
            return;
        }

        for (char c : {'a', 'b', 'c'}) {  // Try 'a', 'b', 'c'
            if (c == prevChar) continue;  // Skip if it repeats

            solve(v, n, st + c, c);  // Recursive call
        }
    }

public:
    string getHappyString(int n, int k) {
        vector<string> v;  // Stores all happy strings
        solve(v, n, "", ' ');  // Start backtracking

        if (v.size() < k) return "";  // If not enough strings, return ""
        return v[k - 1];  // Return the k-th happy string
    }
};
```
### **Why Is This Better?**
✅ **No need for a stack** → We just track the last character directly.  
✅ **No sorting required** → Backtracking generates strings in lexicographical order.  
✅ **Cleaner and faster** → Uses a simple recursive function.  

---

# **Time Complexity Analysis**
- Each character choice leads to **two** valid choices (since the previous character is restricted).  
- The total number of happy strings of length `n` is:  
  \[
  2 \times 2^{(n-1)}
  \]
  - For `n = 3`:  
    \[
    2 \times 2^2 = 2 \times 4 = 8
    \]
  - For `n = 4`:  
    \[
    2 \times 2^3 = 2 \times 8 = 16
    \]
- **Complexity: \( O(2^n) \)** (since we generate all valid strings).  

---

# **Final Thoughts**
1. **Understand the happy string rules**: No consecutive duplicates, only 'a', 'b', 'c'.  
2. **Use backtracking**: Build all valid strings in order.  
3. **Directly return the `k`-th string**: Avoid unnecessary sorting.  
4. **Optimize space**: No need for extra stack structures.  

This approach efficiently generates happy strings and retrieves the correct answer. 🚀