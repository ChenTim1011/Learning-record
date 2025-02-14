[Product of the Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/description/)

## **Problem Explanation**
We need to design a data structure that maintains a sequence of numbers and efficiently retrieves the product of the last **K** numbers.

We need to implement the `ProductOfNumbers` class, which provides the following operations:

1. **`add(int num)`** – Appends the integer `num` to the sequence.
2. **`getProduct(int k)`** – Returns the product of the last `k` numbers in the sequence.

### **Constraints:**
- \(0 \leq num \leq 100\)
- \(1 \leq k \leq 40,000\)
- Up to **40,000** calls will be made to `add()` and `getProduct(k)`.
- The product of any contiguous subsequence of numbers will fit into a **32-bit integer**.

---

## **Example Walkthrough**
```cpp
Input:
["ProductOfNumbers","add","add","add","add","add","getProduct","getProduct","getProduct","add","getProduct"]
[[],[3],[0],[2],[5],[4],[2],[3],[4],[8],[2]]

Output:
[null,null,null,null,null,null,20,40,0,null,32]
```
### **Step-by-Step Execution**
```cpp
ProductOfNumbers productOfNumbers = new ProductOfNumbers();
productOfNumbers.add(3);        // [3]
productOfNumbers.add(0);        // [3, 0]
productOfNumbers.add(2);        // [3, 0, 2]
productOfNumbers.add(5);        // [3, 0, 2, 5]
productOfNumbers.add(4);        // [3, 0, 2, 5, 4]
productOfNumbers.getProduct(2); // Returns 20 → (5 * 4 = 20)
productOfNumbers.getProduct(3); // Returns 40 → (2 * 5 * 4 = 40)
productOfNumbers.getProduct(4); // Returns 0 → (0 * 2 * 5 * 4 = 0)
productOfNumbers.add(8);        // [3, 0, 2, 5, 4, 8]
productOfNumbers.getProduct(2); // Returns 32 → (4 * 8 = 32)
```
---

## **Solution Approach**
### **1. Brute Force Approach (O(k) Query, O(1) Insert)**
The simplest approach is to store all numbers in an array and compute the product of the last `k` numbers by iterating backward in the array. This results in:

- **Insertion (`add()`) → O(1)**
- **Query (`getProduct(k)`) → O(k)**

Since `k` can be as large as **40,000**, this solution is inefficient and may time out.

### **2. Optimized Approach using Prefix Product (O(1) Query)**
To optimize the query time, we use a **prefix product array** where:
- `product[i]` stores the product of all elements from index `0` to `i-1`.
- `getProduct(k)` can be computed efficiently using:
  \[
  \text{product}[n] / \text{product}[n-k]
  \]
  which gives the product of the last `k` elements.

This allows:
- **Insertion (`add()`) → O(1)**
- **Query (`getProduct(k)`) → O(1)**

---

## **Handling Zero Values**
A **zero resets the product** because multiplication by `0` results in `0`. To handle this:
- If `num == 0`, we **reset the prefix product array** because everything before it is irrelevant.
- If `k` extends beyond the last occurrence of `0`, return `0`.

---

## **Implementation (C++ Code)**
```cpp
class ProductOfNumbers {
public:
    vector<int> product = {1}; // Prefix product array, initialized with 1
    int n = 1; // Number of elements stored

    ProductOfNumbers() {
        product.reserve(40000); // Reserve memory to optimize performance
    }
    
    void add(int num) {
        if (num == 0) { 
            product = {1}; // Reset if zero is encountered
            n = 1;
        } else {
            product.push_back(product[n - 1] * num); // Compute prefix product
            n++;
        }
    }
    
    int getProduct(int k) {
        if (n <= k) return 0; // If k exceeds available numbers, result is 0
        return product[n - 1] / product[n - k - 1]; // Compute last k elements' product
    }
};
```

---

## **Time and Space Complexity Analysis**
| Operation | Time Complexity | Space Complexity |
|-----------|---------------|----------------|
| `add(int num)` | **O(1)** | **O(n)** |
| `getProduct(int k)` | **O(1)** | **O(n)** |

This ensures that even with **40,000 operations**, the solution runs efficiently.

---

## **Further Optimizations**
### **1. Using a Static Array**
Instead of using a `vector<int>`, we can use a static array (e.g., `static int product[40000]`) to avoid dynamic memory allocation overhead.

### **2. Reducing Extra Variables**
Instead of maintaining `n`, we can simply use `product.size()` to determine the number of stored elements.

---

## **Summary**
This problem can be efficiently solved using the **prefix product technique**, allowing O(1) operations for both insertion and query. The key takeaways are:

1. **Brute force (O(k) query) is too slow.**
2. **Using prefix product allows O(1) query.**
3. **Handling zeros correctly (resetting the array when encountering zero).**
4. **Using division to quickly retrieve the last K products.**

