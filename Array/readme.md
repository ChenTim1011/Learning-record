## Array
1. Binary Search
- [Binary Search](https://leetcode.com/problems/binary-search/description/)
- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)
- [Search Insert Position](https://leetcode.com/problems/search-insert-position/description/)
- [Sqrt(x)](https://leetcode.com/problems/sqrtx/description/)
- [Sum of Square Numbers](https://leetcode.com/problems/sum-of-square-numbers/description/)
- [Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/description/)

2. Sliding Window
- [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/)
- [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/)
  
3. Two pointers
- [Remove Element](https://leetcode.com/problems/remove-element/description/)
- [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/description/)






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
