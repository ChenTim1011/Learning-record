[Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/)

```c++
class MyQueue {
public:
    MyQueue() {
        // Constructor initializes the queue
    }
    
    stack<int> stackIn;  // Stack to handle incoming elements
    stack<int> stackOut; // Stack to handle outgoing elements

    // Push element x to the back of the queue
    void push(int x) {
        stackIn.push(x);
    }
    
    // Removes the element from in front of the queue and returns that element
    int pop() {
        // If stackOut is empty, transfer all elements from stackIn to stackOut
        if (stackOut.empty()) {
            while (!stackIn.empty()) {
                int y = stackIn.top();
                stackIn.pop();
                stackOut.push(y);
            }
        }
        // Pop the top element from stackOut
        int y = stackOut.top();
        stackOut.pop();
        return y;
    }
    
    // Get the front element
    int peek() {
        // Use pop() to get the front element and then push it back to stackOut
        if (stackOut.empty()) { // Ensure stackOut is not empty
            while (!stackIn.empty()) {
                int y = stackIn.top();
                stackIn.pop();
                stackOut.push(y);
            }
        }
        return stackOut.top();
    }
    
    // Returns whether the queue is empty
    bool empty() {
        // The queue is empty if both stacks are empty
        return stackIn.empty() && stackOut.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

Explanation:
- **push(int x)**: Pushes the element `x` onto `stackIn`.
- **pop()**: If `stackOut` is empty, transfers all elements from `stackIn` to `stackOut` to reverse the order, then pops the top element from `stackOut`.
- **peek()**: Ensures `stackOut` is not empty by transferring elements from `stackIn` if necessary, then returns the top element of `stackOut`.
- **empty()**: Returns `true` if both `stackIn` and `stackOut` are empty, indicating the queue is empty.