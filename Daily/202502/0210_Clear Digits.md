[Clear Digits](https://leetcode.com/problems/clear-digits/description/)

### **Problem Explanation**

You are given a string `s` that consists of **lowercase English letters** and **digits**. The task is to repeatedly perform the following operation **until all digits are removed**:

1. **Delete the first digit** in the string.
2. **Delete the closest non-digit character** to the **left** of this digit.

The final result is the string that remains after removing all digits according to the above operation.

---

### **Key Rules**
1. If a digit is encountered, the first non-digit character to the left of the digit must also be removed.
2. If there are no non-digit characters to the left of the digit, the digit itself is removed.

The input guarantees that it is **always possible** to delete all digits by following this rule.

---

### **Constraints**
- \( 1 \leq s.\text{length} \leq 100 \)
- The string `s` contains only lowercase English letters and digits.
- The input is designed to ensure all digits can be removed.

---

### **Examples**

#### Example 1:
**Input**:  
`s = "abc"`

**Output**:  
`"abc"`

**Explanation**:  
Since there are no digits in the string, no characters are removed, and the original string remains unchanged.

---

#### Example 2:
**Input**:  
`s = "cb34"`

**Output**:  
`""`

**Explanation**:  
1. The first digit is `'3'`. The closest non-digit to the left is `'b'`. Both `'3'` and `'b'` are removed, leaving the string `"c4"`.
2. The next digit is `'4'`. The closest non-digit to the left is `'c'`. Both `'4'` and `'c'` are removed, leaving the string `""`.

---

#### Example 3:
**Input**:  
`s = "a1b2c3"`

**Output**:  
`""`

**Explanation**:  
1. The first digit is `'1'`. The closest non-digit to the left is `'a'`. Remove `'1'` and `'a'`, leaving the string `"b2c3"`.
2. The next digit is `'2'`. The closest non-digit to the left is `'b'`. Remove `'2'` and `'b'`, leaving the string `"c3"`.
3. The next digit is `'3'`. The closest non-digit to the left is `'c'`. Remove `'3'` and `'c'`, leaving the string `""`.

---

### **Better Solution Explanation**

#### **Code**:
```cpp
class Solution {
public:
    string clearDigits(string s) {
        string stack; // A stack-like string to track remaining characters
        for (char c : s) {
            if (isdigit(c)) {
                if (!stack.empty()) {
                    // Remove the closest non-digit character to the left
                    stack.pop_back();
                }
                // The digit itself is not added to the stack, so it's removed
            } else {
                // Add non-digit characters to the stack
                stack.push_back(c);
            }
        }
        return stack; // The remaining characters in the stack are the result
    }
};
```

---

### **Explanation of the Better Solution**

#### **Key Idea**
This solution uses a **stack-like approach** (implemented using a string) to efficiently handle the removal of characters:
- The stack keeps track of **non-digit characters** that have not yet been removed.
- When a digit is encountered, the closest non-digit character to its left is **popped from the stack**, effectively removing it from the result.

#### **Step-by-Step Process**
1. Initialize an empty `stack` string.
2. Iterate through each character in the input string:
   - If the character is a **digit**:
     - Check if the `stack` is not empty.
     - If the `stack` is not empty, pop (remove) the **last character** from the stack. This simulates removing the closest non-digit character to the left of the digit.
     - The digit itself is **not added** to the stack, so it is effectively removed.
   - If the character is a **non-digit**:
     - Push (append) the character onto the stack. This means it remains as a candidate to be removed by subsequent digits.
3. At the end of the loop, the `stack` contains the remaining characters in the string after all digits and their closest non-digit characters have been removed.

#### **Time Complexity**
- **O(n)**: Each character in the string is processed once, and stack operations (`push_back` and `pop_back`) are \( O(1) \).

#### **Space Complexity**
- **O(n)**: The stack can store up to \( n \) characters in the worst case, where \( n \) is the length of the input string.

---

### **Example Walkthrough with Better Solution**

#### Input: `"cb34"`

1. **Initial String**: `"cb34"`  
   - `stack = ""`  

2. **Process 'c'**:  
   - `'c'` is a non-digit, so add it to the stack.  
   - `stack = "c"`

3. **Process 'b'**:  
   - `'b'` is a non-digit, so add it to the stack.  
   - `stack = "cb"`

4. **Process '3'**:  
   - `'3'` is a digit, so remove the last character `'b'` from the stack.  
   - `stack = "c"`

5. **Process '4'**:  
   - `'4'` is a digit, so remove the last character `'c'` from the stack.  
   - `stack = ""`

**Output**: `""`

---

#### **Input: `"a1b2c3"`**

1. **Initial String**: `"a1b2c3"`  
   - `stack = ""`  

2. **Process 'a'**:  
   - `'a'` is a non-digit, so add it to the stack.  
   - `stack = "a"`

3. **Process '1'**:  
   - `'1'` is a digit, so remove `'a'` from the stack.  
   - `stack = ""`

4. **Process 'b'**:  
   - `'b'` is a non-digit, so add it to the stack.  
   - `stack = "b"`

5. **Process '2'**:  
   - `'2'` is a digit, so remove `'b'` from the stack.  
   - `stack = ""`

6. **Process 'c'**:  
   - `'c'` is a non-digit, so add it to the stack.  
   - `stack = "c"`

7. **Process '3'**:  
   - `'3'` is a digit, so remove `'c'` from the stack.  
   - `stack = ""`

**Output**: `""`

---

This solution is efficient, simple, and adheres to the problem constraints. It effectively solves the problem using a stack-based approach.