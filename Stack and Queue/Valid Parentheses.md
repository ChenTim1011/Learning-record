[Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)

## My solution

```c++
class Solution {
public:
    bool isValid(string s) {
        // If the length of the string is odd, it cannot be valid
        if (s.size() % 2 != 0) return false; 
        
        stack<char> st; // Stack to keep track of opening brackets

        for(int i = 0; i < s.size(); i++) {
            // If the character is an opening bracket, push it onto the stack
            if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
                st.push(s[i]);
            } 
            // If the character is a closing bracket, check for matching opening bracket
            else if (s[i] == ')') {
                // Condition 2: Stack is empty or top of stack is not the matching opening bracket
                if (!st.empty() && st.top() == '(') {
                    st.pop(); // Pop the matching opening bracket
                } else {
                    return false; // No matching opening bracket
                }
            } 
            else if (s[i] == ']') {
                // Condition 2: Stack is empty or top of stack is not the matching opening bracket
                if (!st.empty() && st.top() == '[') {
                    st.pop(); // Pop the matching opening bracket
                } else {
                    return false; // No matching opening bracket
                }
            } 
            else if (s[i] == '}') {
                // Condition 2: Stack is empty or top of stack is not the matching opening bracket
                if (!st.empty() && st.top() == '{') {
                    st.pop(); // Pop the matching opening bracket
                } else {
                    return false; // No matching opening bracket
                }
            }
        }

        // Condition 1: After processing all characters, stack should be empty if all brackets are matched
        return st.empty();
    }
};

/*
Explanation of the three conditions:

1. **First Condition**: Left brackets are more than right brackets.
   - After processing the entire string, if the stack is not empty, it means there are unmatched left brackets.
   - Example: "(((" -> Stack will have three '(' left, so return false.

2. **Second Condition**: Bracket types do not match.
   - During the traversal, if a closing bracket is encountered and the stack is empty or the top of the stack is not the matching opening bracket, it means the types do not match.
   - Example: "(]" -> Stack will have '(' and encounter ']', which does not match, so return false.

3. **Third Condition**: Right brackets are more than left brackets.
   - During the traversal, if a closing bracket is encountered and the stack is empty, it means there are more right brackets than left brackets.
   - Example: "())" -> Stack will be empty when encountering the second ')', so return false.

**Valid Case**:
- If the string is processed completely and the stack is empty, it means all brackets are matched correctly.
- Example: "()[]{}" -> Stack will be empty after processing all characters, so return true.
*/
```