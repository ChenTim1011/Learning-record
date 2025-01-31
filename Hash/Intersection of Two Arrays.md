[Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/description/)

### Explanation of the Problem: Intersection of Two Arrays

The task is to find the **intersection** of two integer arrays `nums1` and `nums2`. The **intersection** refers to the common elements present in both arrays. The key points to keep in mind are:

1. **Uniqueness**: The output should only contain unique elements. If an element appears multiple times in both arrays, it should still only appear **once** in the result.
2. **Order**: The order of elements in the result doesn't matter; any valid order is acceptable.
3. **Efficiency**: Since the arrays can have lengths up to 1000, the solution should handle this efficiently.

---

### Examples:

#### Example 1:
Input:  
`nums1 = [1,2,2,1]`, `nums2 = [2,2]`  
Output:  
`[2]`  

Explanation:  
- Both arrays contain the number `2`, so the intersection is `[2]`.  
- Although `2` appears multiple times, we include it only once in the result.

---

#### Example 2:
Input:  
`nums1 = [4,9,5]`, `nums2 = [9,4,9,8,4]`  
Output:  
`[9,4]` (or `[4,9]`)  

Explanation:  
- The numbers `4` and `9` appear in both arrays, so they are part of the intersection.  
- The result can be `[4,9]` or `[9,4]` as the order doesn't matter.

---

### Constraints:
1. `1 <= nums1.length, nums2.length <= 1000`  
   (Both arrays can have up to 1000 elements.)
2. `0 <= nums1[i], nums2[i] <= 1000`  
   (Elements in the arrays are non-negative integers.)

---

### Approach to Solve the Problem:

To efficiently find the intersection:
1. Use a **set** to track unique elements:
   - Use a set for one of the arrays (e.g., `nums1`) to store all its elements.
   - Iterate through the other array (e.g., `nums2`) and check if an element exists in the set.
   - Add the common elements to a result set (ensuring uniqueness).
2. Convert the result set back to a vector to return the final output.

This approach ensures that the solution is efficient, taking advantage of the fast lookup time of sets.

---

### Annotated Code with Comments

```cpp
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> result; // Set to store the unique intersection elements
        unordered_set<int> nums(nums1.begin(), nums1.end()); // Set containing all elements from nums1

        // Iterate through nums2 to check for common elements
        for (auto num : nums2) {
            // If the current element exists in the set from nums1, it's a common element
            if (nums.find(num) != nums.end()) {
                result.insert(num); // Add the element to the result set
            }
        }

        // Convert the result set to a vector and return it
        return vector<int>(result.begin(), result.end());
    }
};
```

---

### Explanation of the Code

1. **Set Initialization**:
   ```cpp
   unordered_set<int> nums(nums1.begin(), nums1.end());
   ```
   - This initializes a set `nums` containing all the unique elements from `nums1`.
   - Using a set ensures that duplicates in `nums1` are ignored automatically.

2. **Finding Common Elements**:
   ```cpp
   for (auto num : nums2) {
       if (nums.find(num) != nums.end()) {
           result.insert(num);
       }
   }
   ```
   - Iterate through each element in `nums2`.
   - Check if the element exists in the set `nums` using `nums.find(num)`.
   - If it exists, insert it into the `result` set (which also ensures uniqueness).

3. **Returning the Result**:
   ```cpp
   return vector<int>(result.begin(), result.end());
   ```
   - The result set, which contains all unique common elements, is converted into a vector and returned.

---

### Example Walkthrough

#### Input:
`nums1 = [4,9,5]`, `nums2 = [9,4,9,8,4]`

#### Step 1: Create a set from `nums1`:
`nums = {4, 9, 5}`

#### Step 2: Iterate through `nums2`:
- For `9`: Found in `nums`, add to `result`.
- For `4`: Found in `nums`, add to `result`.
- For `9`: Already in `result`, skip.
- For `8`: Not in `nums`, skip.
- For `4`: Already in `result`, skip.

#### Step 3: Result Set:
`result = {4, 9}`

#### Step 4: Convert to Vector:
Output: `[4, 9]` (or `[9, 4]`)

---

### Complexity Analysis

1. **Time Complexity**:
   - Creating the set from `nums1`: \(O(n)\), where \(n\) is the size of `nums1`.
   - Iterating through `nums2` and checking each element in the set: \(O(m)\), where \(m\) is the size of `nums2`.
   - Total: \(O(n + m)\).

2. **Space Complexity**:
   - Set `nums` to store elements from `nums1`: \(O(n)\).
   - Set `result` to store the intersection: \(O(min(n, m))\).

