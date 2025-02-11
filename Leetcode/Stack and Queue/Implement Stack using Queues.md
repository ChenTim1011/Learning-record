[Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/description/)

## Intuition method

```c++
class MyStack {
public:
    MyStack() {
        // Constructor initializes the stack
    }

    queue<int> queue1; // Primary queue to hold elements
    queue<int> queue2; // Secondary queue to assist in operations
    
    // Push element x onto stack
    void push(int x) {
        queue1.push(x); // Push the element to the primary queue
    }
    
    // Removes the element on top of the stack and returns that element
    int pop() {
        int q1size = queue1.size(); // Get the size of the primary queue
        int result = queue1.back(); // Get the last element which is the top of the stack
        // Transfer all elements except the last one to the secondary queue
        while (q1size > 1) {
            queue2.push(queue1.front());
            queue1.pop();
            q1size--;
        }
        queue1.pop(); // Remove the last element
        // Transfer elements back to the primary queue
        while (!queue2.empty()) {
            queue1.push(queue2.front());
            queue2.pop();  
        }
        return result; // Return the top element
    }
    
    // Get the top element
    int top() {
        return queue1.back(); // The last element in the primary queue is the top of the stack
    }
    
    // Returns whether the stack is empty
    bool empty() {
        return queue1.empty() ; // The stack is empty if queue1 are empty
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
 ```

# Only one queue 

 ```c++
 class MyStack {
public:
    MyStack() {
        // Constructor initializes the stack
    }

    queue<int> queue1; // Single queue to hold elements
    
    // Push element x onto stack
    void push(int x) {
        queue1.push(x); // Push the element to the queue
    }
    
    // Removes the element on top of the stack and returns that element
    int pop() {
        int q1size = queue1.size(); // Get the size of the queue
        // Transfer all elements except the last one to the back of the queue
        for (int i = 0; i < q1size - 1; i++) {
            queue1.push(queue1.front());
            queue1.pop();
        }
        int result = queue1.front(); // The last element is the top of the stack
        queue1.pop(); // Remove the last element

        return result; // Return the top element
    }
    
    // Get the top element
    int top() {
        return queue1.back(); // The last element in the queue is the top of the stack
    }
    
    // Returns whether the stack is empty
    bool empty() {
        return queue1.empty(); // The stack is empty if the queue is empty
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
 ```

Explanation:
- **Intuition method**: This approach uses two queues. The `push` operation simply adds the element to the primary queue. The `pop` operation transfers all elements except the last one to the secondary queue, removes the last element (which is the top of the stack), and then transfers the elements back to the primary queue. The `top` operation returns the last element of the primary queue. The `empty` operation checks if both queues are empty.
- **Only one queue**: This approach uses a single queue. The `push` operation adds the element to the queue. The `pop` operation transfers all elements except the last one to the back of the queue, removes the last element (which is the top of the stack), and then returns it. The `top` operation returns the last element of the queue. The `empty` operation checks if the queue is empty.

Both methods ensure that the stack operations are implemented using only the standard operations of a queue.