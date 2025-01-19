[Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description/)

## My solution

```c++
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> result; // Initialize a stack to keep track of numbers
        // tokens = ["4","13","5","/","+"]
        // (4 + (13 / 5)) = 6
        for(int i=0; i<tokens.size(); i++) {
            // If we encounter an operator, pop two numbers from the stack to calculate and push the result back to the stack
            if(tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
                int num1 = result.top();
                result.pop();
                int num2 = result.top();
                result.pop();
                if(tokens[i] == "+") {
                    result.push(num2 + num1);
                } else if(tokens[i] == "-") {
                    result.push(num2 - num1);
                } else if(tokens[i] == "*") {
                    result.push(num2 * num1);
                } else if(tokens[i] == "/") {
                    result.push(num2 / num1);
                }
            } else {
                // Convert string to integer using stoi and push it onto the stack
                result.push(stoi(tokens[i]));
            }
        }
        return result.top(); // The final result is the top element of the stack
    }
};
```

Example Explanation:
- Consider the input tokens `["4","13","5","/","+"]`.
- Initialize an empty stack.
- Iterate through each token:
  - '4': It's a number, convert it to integer using `stoi` and push 4 onto the stack. Stack: [4]
  - '13': It's a number, convert it to integer using `stoi` and push 13 onto the stack. Stack: [4, 13]
  - '5': It's a number, convert it to integer using `stoi` and push 5 onto the stack. Stack: [4, 13, 5]
  - '/': It's an operator, pop 5 and 13 from the stack, calculate 13 / 5 = 2, push 2 onto the stack. Stack: [4, 2]
  - '+': It's an operator, pop 2 and 4 from the stack, calculate 4 + 2 = 6, push 6 onto the stack. Stack: [6]
- The final result is the top element of the stack, which is 6.

Explanation of `stoi`:
- The `stoi` function converts a string to an integer. It is used here to convert the string tokens to integers before pushing them onto the stack.
- Using `atoi` would result in an error because `atoi` expects a C-style string (null-terminated character array), not a `std::string`.

Correction of logical error:
- The condition `tokens[i]=='+' || tokens[i]=='-' || tokens[i]=='*' || tokens[i]=='/'` results in a compile error because `tokens[i]` is a `std::string` and cannot be compared to a `char`.
- The correct condition is `tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/"`, which compares `std::string` to `std::string`.

Advantages of Postfix Notation:
- Postfix notation (Reverse Polish Notation) does not require parentheses to denote operator precedence and associativity. This simplifies the parsing process.
- In postfix notation, operators follow their operands, which allows for straightforward evaluation using a stack. Each operator applies to the most recent operands on the stack.
- Postfix notation is more efficient for computers to evaluate because it eliminates the need for complex precedence rules and reduces the number of operations required to parse and evaluate expressions.
- Example: The infix expression `(4 + (13 / 5))` is written as `4 13 5 / +` in postfix notation. This can be evaluated using a stack in a single left-to-right pass without needing to consider operator precedence or parentheses.