[Remove All Occurrences of a Substring](https://leetcode.com/problems/remove-all-occurrences-of-a-substring/description/)

### **Deep Explanation of the Problem**

The task is to repeatedly remove the first occurrence of a substring (`part`) from a string (`s`) until `part` no longer exists in `s`. A **substring** is defined as a contiguous sequence of characters in a string.

To achieve this, we can use:
1. **`find()`**: To locate the starting index of the substring `part` in `s`.
2. **`erase()`**: To remove the characters corresponding to `part` from the string.

---

### **How `find()` and `erase()` Work**

1. **`find()`**:
   - This function searches for the first occurrence of a substring in the string and returns its starting index.
   - If the substring is not found, it returns `std::string::npos`, a constant representing "not found."

   **Example**:
   ```cpp
   std::string s = "hello world";
   size_t index = s.find("world");
   // index = 6 (since "world" starts at position 6 in "hello world")

   size_t not_found = s.find("abc");
   // not_found = std::string::npos (because "abc" does not exist in the string)
   ```

2. **`erase()`**:
   - This function removes characters from the string, starting at a specific index, and deletes a given number of characters.
   - Syntax: `string.erase(start_index, length_to_erase)`.

   **Example**:
   ```cpp
   std::string s = "hello world";
   s.erase(6, 5);
   // s = "hello " (removed 5 characters starting at index 6)
   ```

---

### **Step-by-Step Algorithm**

1. **Initialize the process**:
   - Find the first occurrence of `part` in `s` using `find()`.
   - If found, get its starting index.

2. **Remove `part`**:
   - Use `erase()` to remove `part` from `s`. This involves:
     - Start index: The index returned by `find()`.
     - Length: The length of the substring `part`.

3. **Repeat until `part` no longer exists in `s`**:
   - Continue finding and erasing `part` until `find()` returns `std::string::npos`.

4. **Return the final string**:
   - After all occurrences are removed, return `s`.

---

### **Code Walkthrough**

Here’s the implementation in C++:

```cpp
class Solution {
public:
    string removeOccurrences(string s, string part) {
        size_t index = s.find(part);  // Step 1: Find the first occurrence
        while (index != string::npos) {  // Continue while "part" exists in "s"
            s.erase(index, part.length());  // Step 2: Erase "part" from "s"
            index = s.find(part);  // Step 3: Find the next occurrence of "part"
        }
        return s;  // Step 4: Return the final modified string
    }
};
```

---

### **Example Walkthrough**

#### Example 1: 
Input: `s = "daabcbaabcbc"`, `part = "abc"`

1. Initial `s = "daabcbaabcbc"`.  
   Find `part = "abc"` at index `2`.  
   Remove `"abc"`, resulting in `s = "dabaabcbc"`.

2. `s = "dabaabcbc"`.  
   Find `part = "abc"` at index `4`.  
   Remove `"abc"`, resulting in `s = "dababc"`.

3. `s = "dababc"`.  
   Find `part = "abc"` at index `3`.  
   Remove `"abc"`, resulting in `s = "dab"`.

4. `s = "dab"`.  
   No more occurrences of `part`. Stop.

Output: `"dab"`

---

#### Example 2: 
Input: `s = "axxxxyyyyb"`, `part = "xy"`

1. Initial `s = "axxxxyyyyb"`.  
   Find `part = "xy"` at index `4`.  
   Remove `"xy"`, resulting in `s = "axxxyyyb"`.

2. `s = "axxxyyyb"`.  
   Find `part = "xy"` at index `3`.  
   Remove `"xy"`, resulting in `s = "axxyyb"`.

3. `s = "axxyyb"`.  
   Find `part = "xy"` at index `2`.  
   Remove `"xy"`, resulting in `s = "axyb"`.

4. `s = "axyb"`.  
   Find `part = "xy"` at index `1`.  
   Remove `"xy"`, resulting in `s = "ab"`.

5. `s = "ab"`.  
   No more occurrences of `part`. Stop.

Output: `"ab"`

---

### **Complexity Analysis**

1. **Time Complexity**:
   - `find()` takes **O(n)** in the worst case.
   - `erase()` creates a new string, also **O(n)**.
   - Suppose there are **k** occurrences of `part` in `s`. Each occurrence involves `O(n)` work.
   - Worst-case complexity: **O(n²/m)** where `m` is the length of `part`.

2. **Space Complexity**:
   - The string manipulations result in new copies being created, leading to **O(n)** space usage.

---

### **Key Takeaways**
- `find()` helps locate substrings efficiently.
- `erase()` modifies strings by removing characters.
- The iterative approach repeatedly removes substrings until none remain.
- Optimizing space (e.g., using `StringBuilder` in Java or `std::ostringstream` in C++) can help reduce memory overhead for larger strings.