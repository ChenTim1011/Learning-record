[Two Sum](https://leetcode.com/problems/two-sum/description/)

### **Problem Explanation**

The problem asks us to find two indices of numbers in an array (`nums`) such that the sum of these two numbers equals the given target value. You must return these indices as a pair in any order. The key constraints are:

1. Each input has exactly **one valid solution**.
2. The **same element cannot be used twice** (i.e., each index can be used only once).
3. The returned result can be in any order, as long as it satisfies the condition.

---

### **Examples**

#### Example 1:
- **Input**: `nums = [2,7,11,15], target = 9`
- **Output**: `[0, 1]`
- **Explanation**:  
  The numbers `nums[0] = 2` and `nums[1] = 7` add up to `9` (i.e., `2 + 7 = 9`). Thus, we return `[0, 1]`.

#### Example 2:
- **Input**: `nums = [3,2,4], target = 6`
- **Output**: `[1, 2]`
- **Explanation**:  
  The numbers `nums[1] = 2` and `nums[2] = 4` add up to `6` (i.e., `2 + 4 = 6`). Thus, we return `[1, 2]`.

#### Example 3:
- **Input**: `nums = [3,3], target = 6`
- **Output**: `[0, 1]`
- **Explanation**:  
  The numbers `nums[0] = 3` and `nums[1] = 3` add up to `6` (i.e., `3 + 3 = 6`). Thus, we return `[0, 1]`.

---

### **Constraints**

1. \( 2 \leq \text{nums.length} \leq 10^4 \)  
2. \( -10^9 \leq \text{nums[i]}, \text{target} \leq 10^9 \)  
3. **Exactly one valid solution exists.**

---

### **Solution Concept**

The goal is to find two indices such that the sum of the numbers at those indices equals the target. Instead of using a brute-force approach (which would check all pairs of numbers and have \( O(n^2) \) time complexity), we use a more efficient method with a hash map (`unordered_map` in C++).

**Steps:**
1. Use an `unordered_map` to store the values of `nums` as keys and their indices as values.
2. For each number `nums[i]`, compute its complement:  
   \( \text{complement} = \text{target} - \text{nums}[i] \).
3. Check if the complement exists in the map:
   - If yes, return the indices of the complement (found in the map) and the current index.
   - If no, insert the current number and its index into the map.
4. Repeat until a solution is found.

The hash map ensures \( O(1) \) lookup and insertion, so the overall complexity is \( O(n) \).

---

### **Code with Detailed Explanation**

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // Create an unordered map to store numbers as keys and their indices as values
        unordered_map<int, int> maps;

        // Iterate through the array
        for (int i = 0; i < nums.size(); i++) {
            // Calculate the complement (target - current number)
            int complement = target - nums[i];

            // Check if the complement exists in the map
            auto iter = maps.find(complement);
            if (iter != maps.end()) {
                // If found, return the indices of the complement and the current number
                return {iter->second, i};
            } 

            // If not found, add the current number and its index to the map
            maps.insert(pair<int, int>(nums[i], i));
        }

        // Return an empty vector if no solution is found (should not happen as per constraints)
        return {};
    }
};
```

---

### **How the Code Works**

1. **Initialization**:  
   An `unordered_map` (`maps`) is created to store the mapping of each number in `nums` to its index.  

2. **Iteration**:
   - For each index `i` in `nums`, calculate the complement of `nums[i]`:
     \( \text{complement} = \text{target} - \text{nums}[i] \).
   - Check if this complement exists in the map.  
     - If it exists, we know the current number and the complement add up to the target, so return their indices.
     - If it does not exist, store the current number and its index in the map for future reference.

3. **Return**:
   - The function will always return a solution because the problem guarantees exactly one valid answer.

---

### **Example Walkthrough**

#### Input: `nums = [2, 7, 11, 15], target = 9`

1. **Iteration 1 (i = 0)**:
   - Current number: `nums[0] = 2`.
   - Complement: \( 9 - 2 = 7 \).
   - `maps` is empty, so insert `{2: 0}` into the map.

2. **Iteration 2 (i = 1)**:
   - Current number: `nums[1] = 7`.
   - Complement: \( 9 - 7 = 2 \).
   - Check if `2` exists in `maps`:
     - Yes, `2` exists with index `0`.
   - Return `[0, 1]`.

#### Output: `[0, 1]`

---

### **Complexity Analysis**

1. **Time Complexity**:
   - **Hash Map Lookup and Insertion**: Both are \( O(1) \) on average.
   - **Total Iterations**: \( O(n) \), where \( n \) is the size of the `nums` array.
   - Overall: \( O(n) \).

2. **Space Complexity**:
   - We use an `unordered_map` to store at most \( n \) elements, so the space complexity is \( O(n) \).

---

### **Final Notes**
This efficient solution leverages the power of hash maps to achieve \( O(n) \) time complexity, making it suitable for large input sizes within the constraints. The use of complements ensures that every number is checked only once, and no unnecessary computations are performed.