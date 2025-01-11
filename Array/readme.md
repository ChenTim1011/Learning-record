
### Key Points about Arrays:

1. **Array Characteristics**:
   - Arrays are a collection of data of the same type stored in **contiguous memory spaces**.
   - Elements in an array are accessed via **indices starting at 0**.
   - **Adding or removing elements** requires shifting other elements in memory.

2. **C++ vs Java Two-Dimensional Arrays**:
   - In **C++**, two-dimensional arrays are stored in **contiguous memory**. This can be verified by inspecting the memory addresses, where adjacent elements are spaced based on their type size (e.g., 4 bytes for `int`).
   - In **Java**, two-dimensional arrays are not contiguous. Each row is stored as a separate object with non-contiguous memory addresses. Addressing is handled by the **Java Virtual Machine**.

3. **C++ Specifics**:
   - C++'s `vector` is a container built on an array, but it offers additional functionality like dynamic resizing.
   - Elements in an array cannot be physically deleted, only **overwritten**.

4. **Practical Differences**:
   - Contiguous memory in C++ allows for better **cache performance** and direct memory management.
   - Java abstracts memory details, prioritizing ease of use and flexibility over performance in low-level memory operations.



---

### Comparison of `vector` and `array` in C++:

| **Feature**          | **Array**                                         | **Vector**                                      |
|-----------------------|--------------------------------------------------|------------------------------------------------|
| **Size**             | Fixed at compile-time.                           | Dynamic, can resize during runtime.            |
| **Memory**           | Contiguous memory allocation.                    | Contiguous memory allocation (may reallocate). |
| **Performance**      | Faster due to no overhead of resizing.           | Slightly slower due to dynamic resizing.       |
| **Ease of Use**      | Manual memory management for insertion/deletion. | Built-in methods for insertion and deletion.   |
| **Safety**           | No bounds checking (undefined behavior if exceeded). | Bounds checking with `at()` method.          |
| **Flexibility**      | Rigid, fixed size, minimal flexibility.          | Highly flexible due to dynamic resizing.       |
| **Methods/Utilities**| None (except low-level pointer arithmetic).      | Rich set of methods (e.g., `push_back`, `pop_back`). |
| **Use Case**         | Suitable for fixed-size, low-level tasks.        | Preferred for dynamic-sized collections.       |

---

### When to Use:
- **Array:** Use when you need fixed-size data storage with high performance and minimal overhead.
- **Vector:** Use when you need dynamic resizing, flexibility, and ease of use with built-in utilities.

reference:
[leetcode-master](https://github.com/youngyangyang04/leetcode-master)

Intuition
We want to find the square root of a given non-negative integer x. Instead of using a traditional approach like repeatedly subtracting numbers until we reach 0 or using a library function, we'll use a smarter method called "Binary Search." Binary Search helps us quickly find the square root by repeatedly narrowing down the search range.

Approach
We first check if x is 0 or 1. If it is, we know that the square root of 0 and 1 is 0 and 1 respectively, so we directly return x.

For any other value of x, we set up a search range between 1 and x. We initialize two variables start and end to represent the range.

Now comes the clever part: We use a while loop to repeatedly divide the search range in half (Binary Search) to find the square root.

In each iteration of the loop, we calculate the middle value mid using the formula start + (end - start) / 2. This formula ensures that we don't encounter any integer overflow when dealing with large values of x.

Next, we calculate the square of mid and compare it with x.

If the square of mid is greater than x, we know the square root lies in the lower half of the search range. So, we move the end pointer to the left to narrow down the search range.

If the square of mid is equal to x, we have found the square root! So, we return mid as the answer.

If the square of mid is less than x, we know the square root lies in the upper half of the search range. So, we move the start pointer to the right to continue the search.

We repeat steps 4 to 8 until the start pointer becomes greater than the end pointer. At this point, we have found the floor value of the square root, and end holds that value.

To ensure that we return the correct floor value of the square root, we round down the value of end to the nearest integer using the Math.round() method.

Complexity
Time complexity:
The time complexity of this approach is O(logN). It's very efficient because Binary Search reduces the search range by half in each iteration, making the search faster.

Space complexity:
The space complexity is O(1), which means the amount of extra memory used is constant, regardless of the input. We only use a few variables to store the search range and the middle value during the computation.

```c++
class Solution {
public:
    int mySqrt(int x) {
        // For special cases when x is 0 or 1, return x.
        if (x == 0 || x == 1)
            return x;
        
        // Initialize the search range for the square root.
        int start = 1;
        int end = x;
        int mid = -1;
        
        // Perform binary search to find the square root of x.
        while (start <= end) {
            // Calculate the middle point using "start + (end - start) / 2" to avoid integer overflow.
            mid = start + (end - start) / 2;
            
            // Convert mid to long to handle large values without overflow.
            long long square = static_cast<long long>(mid) * mid;
            
            // If the square of the middle value is greater than x, move the "end" to the left (mid - 1).
            if (square > x)
                end = mid - 1;
            else if (square == x)
                // If the square of the middle value is equal to x, we found the square root.
                return mid;
            else
                // If the square of the middle value is less than x, move the "start" to the right (mid + 1).
                start = mid + 1;
        }
        
        // The loop ends when "start" becomes greater than "end", and "end" is the integer value of the square root.
        // However, since we might have been using integer division in the calculations,
        // we round down the value of "end" to the nearest integer to get the correct square root.
        return static_cast<int>(std::round(end));
    }
};
```