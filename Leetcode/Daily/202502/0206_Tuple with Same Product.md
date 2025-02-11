[Tuple with Same Product](https://leetcode.com/problems/tuple-with-same-product/description/)




## Problem Recap

You are given an array of distinct positive integers, and you need to count all 4-tuples `(a, b, c, d)` such that:

- **a \* b = c \* d**  
- **a, b, c, d** are all distinct elements from the array.

For example, if the input is `nums = [2, 3, 4, 6]`, one valid 4-tuple is `(2,6,3,4)`. Notice that the ordering matters: the tuple `(2,6,3,4)` is considered different from `(6,2,3,4)`, and each matching pair can be arranged in 8 different ways (as explained below).

---

## Key Insight

Checking all possible 4-tuples directly would result in an O(n⁴) solution, which is impractical for arrays of size up to 1000. Instead, we can break the problem down by:

1. **Pairing the numbers:**  
   We split the 4 numbers into two pairs. Suppose we have two pairs `(a, b)` and `(c, d)`. The condition then becomes:
   \[
   a \times b = c \times d
   \]

2. **Hashing Pair Products:**  
   Instead of checking all quadruplets, we can compute the product for every unique pair `(i, j)` (where `i < j`) and store the frequency of each product in a hash map (or dictionary).  
   - **Key:** The product value.
   - **Value:** The number of pairs that have produced that product so far.

---

## Detailed Approach

### 1. Enumerate All Pairs

- Use a nested loop where the outer loop runs with index `i` and the inner loop runs with index `j` (with `j > i`) to ensure that each pair is unique.
- For each pair `(nums[i], nums[j])`, calculate the product:
  \[
  \text{product} = nums[i] \times nums[j]
  \]

### 2. Count Matching Pairs Using a Hash Map

- For every pair, check if the calculated product already exists in the hash map.
- If it does, that means there are some previously encountered pairs with the same product.
- **Important:** Each time you find a new pair with an already seen product, it can combine with each of the existing pairs to form valid 4-tuples.

### 3. Why Multiply by 8?

Assume you have two pairs:
- Pair 1: \((p_1, p_2)\)
- Pair 2: \((q_1, q_2)\)
  
Both pairs satisfy:
\[
p_1 \times p_2 = q_1 \times q_2
\]

There are 8 valid ways to arrange these numbers into a 4-tuple:
- You can choose either pair as the first or second part of the tuple.
- Within each pair, the order of the numbers can be swapped.

The possible arrangements are:
1. \((p_1, p_2, q_1, q_2)\)
2. \((p_2, p_1, q_1, q_2)\)
3. \((p_1, p_2, q_2, q_1)\)
4. \((p_2, p_1, q_2, q_1)\)
5. \((q_1, q_2, p_1, p_2)\)
6. \((q_2, q_1, p_1, p_2)\)
7. \((q_1, q_2, p_2, p_1)\)
8. \((q_2, q_1, p_2, p_1)\)

Thus, when a new pair is found with a product that already exists `count` times, you can form `8 * count` new valid tuples.

### 4. Updating the Hash Map

After using the current pair to update the answer, increment the count for the product in the hash map. This ensures that future pairs encountering the same product will combine with all previous pairs.

---

## Complexity Analysis

- **Time Complexity:**  
  The approach involves iterating over all pairs in the array. With two nested loops, the complexity is O(n²).

- **Space Complexity:**  
  In the worst case, the hash map may store up to O(n²) distinct products (if every pair results in a different product).

---

## Example Walkthrough

Consider the array:  
`nums = [1, 2, 3, 4, 6, 8, 12, 24]`

Focusing on pairs that yield a product of 24:

1. **Pair (1, 24):**  
   - Product: 24  
   - The hash map does not yet contain 24.  
   - Add `(1, 24)` to the map: `mp[24] = 1`.

2. **Pair (2, 12):**  
   - Product: 24  
   - The hash map already has one pair with product 24.  
   - Form new tuples: `8 * 1 = 8` new valid 4-tuples.  
   - Update map: `mp[24] = 2`.

3. **Pair (3, 8):**  
   - Product: 24  
   - There are already 2 pairs with product 24.  
   - Form new tuples: `8 * 2 = 16` new valid 4-tuples.  
   - Update map: `mp[24] = 3`.

4. **Pair (4, 6):**  
   - Product: 24  
   - There are already 3 pairs with product 24.  
   - Form new tuples: `8 * 3 = 24` new valid 4-tuples.  
   - Update map: `mp[24] = 4`.

Adding the counts for product 24 gives:  
\[
8 + 16 + 24 = 48 \text{ valid 4-tuples from product 24 alone.}
\]

---

## Code Implementation (C++ Example)

Here is a C++ implementation of the above approach:

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int tupleSameProduct(vector<int>& nums) {
        unordered_map<int, int> mp;
        int ans = 0, n = nums.size();
        
        // Iterate over all unique pairs (i, j) with i < j
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int product = nums[i] * nums[j];
                // If this product was seen before, each previous pair contributes 8 new tuples
                ans += 8 * mp[product];
                // Update the map with the current pair's product
                mp[product]++;
            }
        }
        return ans;
    }
};
```

---

## Summary

1. **Pair the numbers:**  
   By splitting the 4-tuple into two pairs, we only need to compare their products.

2. **Use a hash map:**  
   The hash map efficiently stores the frequency of each pair product, reducing the need for a brute force O(n⁴) approach.

3. **Account for ordering:**  
   Each combination of two pairs yields 8 distinct 4-tuples due to the possible orders within and between pairs.

4. **Overall complexity:**  
   The approach runs in O(n²) time with O(n²) space in the worst case, making it efficient for the problem's constraints.

