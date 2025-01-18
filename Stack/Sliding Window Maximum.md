[Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/)

## Using Monotonic Queue

```c++
class Solution {

private:
    // Define a custom queue class that maintains elements in a monotonic decreasing order
    class MyQueue {
        public:
            deque<int> queue; // Use a deque to store elements
        
            // Remove the front element if it matches the value to be removed
            void pop(int val) {
                if(!queue.empty() && val == queue.front()) {
                    queue.pop_front();
                }
            }

            // Add a new element to the queue while maintaining the monotonic decreasing order
            void push(int val) {
                while(!queue.empty() && val > queue.back()) {
                    queue.pop_back();
                }
                queue.push_back(val);
            }

            // Get the maximum value, which is the front element of the queue
            int front() {
                return queue.front();
            }
    };

public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        MyQueue que; // Initialize the custom monotonic queue
        vector<int> result; // Initialize a vector to store the result

        // Process the first k elements to initialize the window
        for(int i = 0; i < k; i++) {
            que.push(nums[i]);
        }
        result.push_back(que.front()); // Add the maximum value of the first window to the result

        // Process the remaining elements
        for(int i = k; i < nums.size(); i++) {
            que.pop(nums[i - k]); // Remove the element that is out of the current window
            que.push(nums[i]); // Add the new element to the current window
            result.push_back(que.front()); // Add the maximum value of the current window to the result
        }

        return result; // Return the result vector containing the maximum values of each window
    }
};
```

Detailed Explanation:
- This is a classic problem that uses a monotonic queue.
- The challenge is to find the maximum value in a sliding window of size `k`.
- A brute force approach would be to iterate through each window and find the maximum value, resulting in an `O(n * k)` time complexity.
- Using a max-heap (priority queue) is not suitable because it cannot efficiently remove elements that are out of the current window.
- Instead, we use a custom queue that maintains elements in a monotonic decreasing order. This allows us to efficiently get the maximum value of the current window.

Monotonic Queue:
- The custom queue maintains elements in a monotonic decreasing order, with the maximum value at the front.
- When the window slides, we remove the element that is out of the current window and add the new element.
- The `pop` method removes the front element if it matches the value to be removed.
- The `push` method adds a new element while maintaining the monotonic decreasing order by removing elements from the back that are smaller than the new element.
- The `front` method returns the maximum value, which is the front element of the queue.

Example:
- Consider the input `nums = [1,3,-1,-3,5,3,6,7]` and `k = 3`.
- Initialize the queue with the first `k` elements: [1, 3, -1]. The queue maintains [3, -1] (monotonic decreasing).
- The maximum value of the first window is 3.
- Slide the window to the right:
  - Remove 1, add -3. The queue maintains [3, -1, -3]. The maximum value is 3.
  - Remove 3, add 5. The queue maintains [5]. The maximum value is 5.
  - Remove -1, add 3. The queue maintains [5, 3]. The maximum value is 5.
  - Remove -3, add 6. The queue maintains [6]. The maximum value is 6.
  - Remove 5, add 7. The queue maintains [7]. The maximum value is 7.
- The result is [3, 3, 5, 5, 6, 7].