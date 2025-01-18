[Remove All Adjacent Duplicates In String](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/description/)

## Using stack

```c++
class Solution {
public:
    string removeDuplicates(string s) {
        stack<char> sk; // Initialize a stack to keep track of characters
        // s = "abbaca"
        for(int i=0; i<s.size(); i++) {
            // If the stack is empty or the top of the stack is not equal to the current character
            if(sk.empty() || sk.top() != s[i]) {
                sk.push(s[i]); // Push the current character onto the stack
            } else {
                sk.pop(); // If the top of the stack is equal to the current character, pop the stack
            }
        }
        // c a(top)
        string result = ""; // Initialize an empty string to store the result
        while(!sk.empty()) {
            result += sk.top(); // Append the top character of the stack to the result string
            sk.pop(); // Pop the stack
        }
        // a c
        reverse(result.begin(), result.end()); // Reverse the result string to get the correct order
        // c a
        return result; // Return the result string
    }
};
```

## String method without stack

```c++
class Solution {
public:
    string removeDuplicates(string s) {
        string result = ""; // Initialize an empty string to store the result
        // s = "abbaca"  
        for(int i=0; i<s.size(); i++) {
            // If the result string is empty or the last character of the result string is not equal to the current character
            if(result.empty() || result.back() != s[i]) {
                result.push_back(s[i]); // Append the current character to the result string
            } else {
                result.pop_back(); // If the last character of the result string is equal to the current character, remove the last character
            }
        }

        return result; // Return the result string
    }
};
```

Example Explanation:
- Consider the input string `s = "abbaca"`.
- Using the stack method:
  - Initialize an empty stack.
  - Iterate through each character in the string:
    - 'a': Stack is empty, push 'a' onto the stack. Stack: ['a']
    - 'b': Top of the stack is 'a', push 'b' onto the stack. Stack: ['a', 'b']
    - 'b': Top of the stack is 'b', pop the stack. Stack: ['a']
    - 'a': Top of the stack is 'a', pop the stack. Stack: []
    - 'c': Stack is empty, push 'c' onto the stack. Stack: ['c']
    - 'a': Top of the stack is 'c', push 'a' onto the stack. Stack: ['c', 'a']
  - The stack now contains ['c', 'a']. Pop each element and append to the result string, then reverse the result string to get "ca".
- Using the string method:
  - Initialize an empty result string.
  - Iterate through each character in the string:
    - 'a': Result string is empty, append 'a'. Result: "a"
    - 'b': Last character in result is 'a', append 'b'. Result: "ab"
    - 'b': Last character in result is 'b', remove 'b'. Result: "a"
    - 'a': Last character in result is 'a', remove 'a'. Result: ""
    - 'c': Result string is empty, append 'c'. Result: "c"
    - 'a': Last character in result is 'c', append 'a'. Result: "ca"
  - The final result string is "ca".

