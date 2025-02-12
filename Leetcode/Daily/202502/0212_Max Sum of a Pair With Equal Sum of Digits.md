[Max Sum of a Pair With Equal Sum of Digits](https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/description/)


### **Understanding the Problem**
We are given an array `nums` of positive integers. We need to find two different numbers whose **sum of digits** is the same and return the maximum sum of such a pair. If no such pair exists, we return `-1`.

#### **Example Walkthrough**
##### **Example 1**
**Input:**  
`nums = [18, 43, 36, 13, 7]`

1. Compute the sum of digits for each number:
   - `18` → `1 + 8 = 9`
   - `43` → `4 + 3 = 7`
   - `36` → `3 + 6 = 9`
   - `13` → `1 + 3 = 4`
   - `7`  → `7`

2. Group numbers by their digit sums:
   - **Sum 9:** `{18, 36}`
   - **Sum 7:** `{43, 7}`
   - **Sum 4:** `{13}` (No pair)
   - **Sum 7:** `{43, 7}` (Pair found)

3. Compute the maximum sum:
   - `(18, 36) → 18 + 36 = 54`
   - `(43, 7) → 43 + 7 = 50`
   - **Maximum sum = 54**

**Output:** `54`

##### **Example 2**
**Input:**  
`nums = [10, 12, 19, 14]`

1. Compute digit sums:
   - `10` → `1`
   - `12` → `3`
   - `19` → `10`
   - `14` → `5`

2. Since no two numbers share the same digit sum, we return `-1`.

**Output:** `-1`

---

### **Approach and Thought Process**
We need an efficient way to:
1. **Group numbers by their sum of digits**.
2. **Track the largest number for each digit sum**.
3. **Find the maximum possible sum for each group**.

#### **Optimal Approach**
1. Use an **array `mp[82]`** where:
   - `mp[sum]` stores the **largest** number encountered for a particular digit sum.
   - The maximum digit sum possible is `81` (`999999999 → 9+9+9+9+9+9+9+9+9 = 81`), so we use an array of size `82` (indices `0-81`).
   
2. **Iterate through `nums`**, and for each number:
   - Compute its **sum of digits**.
   - Check if a number with the same digit sum has already been seen:
     - If yes, update the maximum pair sum.
     - If no, store it as the largest number for that digit sum.
   - Update `mp[sum]` to always store the **largest number** encountered for this sum.

3. **Return the maximum sum found** (or `-1` if no valid pairs exist).

---

### **Code Implementation**
```cpp
class Solution {
public:
    int maximumSum(vector<int>& nums) {
        int mp[82];  // Stores the largest number for each digit sum
        memset(mp, -1, sizeof(mp));  // Initialize all values to -1
        int ans = -1;

        for (int num : nums) {
            int sumDigits = 0, temp = num;

            // Compute the sum of digits
            while (temp) {
                sumDigits += temp % 10;
                temp /= 10;
            }

            // Check if there is an existing number with the same digit sum
            if (mp[sumDigits] != -1)
                ans = max(ans, num + mp[sumDigits]); // Update max sum if a valid pair is found

            // Store the largest number encountered for this digit sum
            mp[sumDigits] = max(mp[sumDigits], num);
        }

        return ans;
    }
};
```

---

### **Complexity Analysis**
| **Operation** | **Time Complexity** | **Reasoning** |
|--------------|-------------------|--------------|
| Compute sum of digits | O(log N) | Each number has at most 10 digits (`log(10⁹) ≈ 10`) |
| Iterating over `nums` | O(N) | We process each number once |
| Total Complexity | O(N) | Since `log N` is a small constant factor |

- **Space Complexity:** `O(1)`  
  - We use a fixed-size array of `82` elements, which is constant.

---

### **Step-by-Step Execution Example**
#### **Input:** `nums = [18, 43, 36, 13, 7]`

| Index | Number | Sum of Digits | Existing Max in `mp` | New Max Pair Sum | Updated Max |
|-------|--------|--------------|----------------------|------------------|-------------|
| 0     | 18     | 9            | -1                   | -                | 18 ✅        |
| 1     | 43     | 7            | -1                   | -                | 43 ✅        |
| 2     | 36     | 9            | 18                   | 18 + 36 = 54     | 36 ✅        |
| 3     | 13     | 4            | -1                   | -                | 13 ✅        |
| 4     | 7      | 7            | 43                   | 43 + 7 = 50      | 43 (unchanged) |

- **Valid pairs found:**
  - `(18, 36) → 18 + 36 = 54`
  - `(43, 7) → 43 + 7 = 50`
- **Maximum sum:** `54`

#### **Final Output:** `54`

---

### **Edge Cases Considered**
1. **No valid pairs**  
   - Example: `[10, 12, 19, 14]`  
   - Each number has a unique sum of digits → Return `-1`.

2. **All numbers have the same digit sum**  
   - Example: `[99, 81, 18]`  
   - `99 (9+9=18)`, `81 (8+1=9)`, `18 (1+8=9)`
   - Pair `(99, 81)` → Max sum = `99 + 81 = 180`.

3. **Only one element**  
   - Example: `[5]`  
   - No pairs possible → Return `-1`.

4. **Large numbers in input**  
   - Example: `[999999999, 899999999]`  
   - Sum of digits for both = `81`  
   - Valid pair found.

---

### **Summary**
✅ **Efficient O(N) approach** using a fixed-size array.  
✅ **Uses sum of digits as a hash key** to track the largest numbers.  
✅ **Handles all edge cases** including no valid pairs.  
✅ **Memory efficient** (O(1) space complexity).

