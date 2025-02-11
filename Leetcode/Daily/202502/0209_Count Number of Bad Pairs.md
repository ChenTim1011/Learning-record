[Count Number of Bad Pairs](https://leetcode.com/problems/count-number-of-bad-pairs/description/)

### Problem Explanation

The problem asks us to count the number of "bad pairs" in a given array `nums`. A pair `(i, j)` is a **bad pair** if the following condition is NOT met:

\[ j - i = nums[j] - nums[i] \]

Rearranging this equation, we get:

\[ nums[j] - j = nums[i] - i \]

Let us define \( d[i] = nums[i] - i \). This simplifies the problem:
- A pair `(i, j)` is **good** if \( d[i] = d[j] \).
- A pair `(i, j)` is **bad** if \( d[i] \neq d[j] \).

The task is to count all bad pairs in the array.

---

### Key Observations

1. **Total Pairs in the Array**:
   For an array of size \( n \), the total number of pairs is:
   \[
   \text{Total Pairs} = \frac{n \cdot (n - 1)}{2}
   \]

2. **Good Pairs**:
   If a value \( d \) appears \( k \) times in the array, the number of good pairs it contributes is:
   \[
   \text{Good Pairs for } d = \frac{k \cdot (k - 1)}{2}
   \]

3. **Bad Pairs**:
   The total number of bad pairs is:
   \[
   \text{Bad Pairs} = \text{Total Pairs} - \text{Sum of Good Pairs}
   \]

Alternatively, while iterating through the array, we can use a frequency map to track occurrences of \( d \). For each index \( i \), the count of good pairs involving index \( i \) is equal to the current count of \( d[i] \) in the map. Subtract this count from the total pairs as we process each element.

---

### Step-by-Step Algorithm

1. Calculate \( d[i] \) for each element in the array.
2. Use a map to store the frequency of \( d[i] \) values as we iterate.
3. For each index \( i \):
   - Subtract the count of good pairs involving \( d[i] \) (from the map) from the total pairs.
   - Update the frequency of \( d[i] \) in the map.
4. Return the remaining count of bad pairs.

---

### Code with Detailed Comments

Here is the C++ solution with thorough explanations in the comments:

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    long long countBadPairs(vector<int>& nums) {
        int n = nums.size();
        
        // Step 1: Calculate the total number of pairs
        long long totalPairs = (long long)n * (n - 1) / 2;
        
        // Step 2: Create a map to track the frequency of `d[i] = nums[i] - i`
        unordered_map<int, long long> diffCount;
        
        // Step 3: Iterate through the array
        for (int i = 0; i < n; i++) {
            // Calculate the difference `d[i]`
            int diff = nums[i] - i;
            
            // Subtract the number of good pairs for the current `diff`
            totalPairs -= diffCount[diff];
            
            // Increment the frequency of the current `diff`
            diffCount[diff]++;
        }
        
        // Step 4: Return the remaining total pairs, which are bad pairs
        return totalPairs;
    }
};
```

---

### Walkthrough of Example

#### Example 1: \( \text{nums} = [4, 1, 3, 3] \)

1. Compute \( d[i] = nums[i] - i \):
   - \( d[0] = 4 - 0 = 4 \)
   - \( d[1] = 1 - 1 = 0 \)
   - \( d[2] = 3 - 2 = 1 \)
   - \( d[3] = 3 - 3 = 0 \)

   So, \( d = [4, 0, 1, 0] \).

2. Total pairs:
   \[
   \text{Total Pairs} = \frac{4 \cdot 3}{2} = 6
   \]

3. Track \( d[i] \) using the map:
   - For \( i = 0 \): \( d[0] = 4 \), no good pairs yet. Add \( 4 \) to the map.
   - For \( i = 1 \): \( d[1] = 0 \), no good pairs yet. Add \( 0 \) to the map.
   - For \( i = 2 \): \( d[2] = 1 \), no good pairs yet. Add \( 1 \) to the map.
   - For \( i = 3 \): \( d[3] = 0 \), 1 good pair with \( d[1] = 0 \). Subtract 1 from total pairs.

4. Final result:
   \[
   \text{Bad Pairs} = 6 - 1 = 5
   \]

---

### Complexity Analysis

1. **Time Complexity**:
   - Calculating \( d[i] \) and updating the map takes \( O(n) \).
   - Overall: \( O(n) \).

2. **Space Complexity**:
   - The map stores at most \( n \) unique \( d[i] \) values: \( O(n) \).

This is efficient for the constraints \( 1 \leq n \leq 10^5 \).