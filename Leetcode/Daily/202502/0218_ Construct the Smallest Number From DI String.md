[ Construct the Smallest Number From DI String](https://leetcode.com/problems/construct-smallest-number-from-di-string/description/)


### **Understanding the Problem**  
You are given a **0-indexed string** `pattern` of length `n`, consisting only of the characters:  
- `'I'` (Increasing): which means `num[i] < num[i+1]`
- `'D'` (Decreasing): which means `num[i] > num[i+1]`  

Your task is to construct a **0-indexed string `num` of length `n+1`**, which satisfies the following conditions:  
1. `num` consists of digits **'1' to '9'**, where each digit is **used at most once**.  
2. The relationships described in `pattern` are followed (i.e., increasing when `'I'`, decreasing when `'D'`).  
3. Among all valid numbers that satisfy the conditions, you need to return the **lexicographically smallest one**.  

---

### **Examples**  

#### **Example 1**  
**Input:**  
```cpp
pattern = "IIIDIDDD"
```
**Output:**  
```cpp
"123549876"
```
**Explanation:**  
- `pattern = "IIIDIDDD"`
- We must have increasing (`I`) at indices `[0, 1, 2, 4]` → `num[i] < num[i+1]`  
- We must have decreasing (`D`) at indices `[3, 5, 6, 7]` → `num[i] > num[i+1]`  
- Some possible valid numbers: `"245639871"`, `"135749862"`, `"123849765"`  
- The **smallest lexicographical order** that satisfies all conditions is `"123549876"`  

---

#### **Example 2**  
**Input:**  
```cpp
pattern = "DDD"
```
**Output:**  
```cpp
"4321"
```
**Explanation:**  
- `pattern = "DDD"`
- We must have `num[0] > num[1] > num[2] > num[3]`
- Some possible values: `"9876"`, `"7321"`, `"8742"`  
- The **smallest lexicographical order** is `"4321"`.  

---

### **Constraints**
- `1 <= pattern.length <= 8`
- `pattern` consists of only `'I'` and `'D'`.  

---

### **Approach: Greedy + Reverse Substring**  

To construct the **lexicographically smallest** number while satisfying the given constraints, we use a **greedy approach** combined with **substring reversal**.

#### **Step 1: Generate Initial Sequence**
We initialize a **sorted** number sequence `"123...n+1"` so that we start with the smallest lexicographical order.  

#### **Step 2: Compute Successive `D` Count (`succD` Array)**
Define an array `succD[i]` that represents **how many consecutive `D` characters start from index `i`**.  
- If `pattern[i] == 'D'`, then `succD[i] = succD[i+1] + 1`
- Otherwise, `succD[i] = 0`  

This helps us efficiently determine the range of numbers that need to be **reversed** in order to satisfy the decreasing condition.

#### **Step 3: Reverse Segments Corresponding to `D`**
For each `D` in `pattern`, reverse the **smallest necessary segment** of the number sequence to make the numbers decrease.  
The segment to reverse starts from `i` and spans `succD[i] + 1` elements.

---

### **Code Implementation (C++)**
```cpp
class Solution {
public:
    string smallestNumber(string& pattern) {
        const int n = pattern.size();
        vector<char> succD(n, 0); // Store the count of consecutive 'D' from index i

        // Compute consecutive 'D' counts
        succD.back() += (pattern.back() == 'D');  
        for (int i = n - 2; i >= 0; i--) {
            succD[i] = (pattern[i] == 'D') ? succD[i + 1] + 1 : 0;
        }

        // Initialize the smallest lexicographical order string "123...n+1"
        string ans(n + 1, ' ');
        iota(ans.begin(), ans.end(), '1'); 

        // Process the pattern and reverse the necessary segments
        for (int i = 0; i < n; i++) {
            if (pattern[i] == 'D') {
                // Reverse from ans[i] to ans[i + succD[i]]
                reverse(ans.begin() + i, ans.begin() + i + 1 + succD[i]);
            }
            // Skip the already processed 'D' segments
            i += succD[i];
        }
        return ans;
    }
};
```

---

### **Detailed Explanation of Code**
#### **Step 1: Compute `succD[]`**
We iterate **backwards** through `pattern`, computing how many consecutive `D`s start at each index.  

Example:  
For `pattern = "IIIDIDDD"`, we get:  
```cpp
succD = [0, 0, 0, 1, 0, 3, 2, 1]
```
This means:
- `succD[3] = 1` → A single `D` starts at index `3`
- `succD[5] = 3` → Three `D`s start at index `5`
  
This helps determine the correct range to reverse later.

#### **Step 2: Construct Initial Number String**
We create an ordered string `"123456789"`, ensuring the smallest lexicographical order.

#### **Step 3: Reverse Decreasing Segments**
When encountering a `D`, we **reverse** the segment `[i, i + succD[i]]` to enforce the decreasing relationship.

Example for `pattern = "IIIDIDDD"`:
1. Initial `"123456789"`
2. Reverse at `i = 3` (1 `D` at `succD[3]`):  
   - `"123546789"`
3. Reverse at `i = 5` (3 `D`s at `succD[5]`):  
   - `"123549876"`

#### **Step 4: Skip Processed `D` Segments**
`i += succD[i]` ensures that we skip over segments that have already been processed.

---

### **Complexity Analysis**
- **Time Complexity: O(n)**  
  - Computing `succD` takes **O(n)**
  - Constructing `ans` takes **O(n)**
  - Reversing segments takes **O(n)**
  - Total complexity: **O(n)**
  
- **Space Complexity: O(n)**  
  - `succD` array and `ans` string each take **O(n)** space.

---

### **Summary**
1. **Generate `"123...n+1"`** to ensure lexicographical order.
2. **Compute `succD[i]`**: track consecutive `D`s.
3. **Reverse segments for `D` regions** to satisfy constraints.
4. **Skip processed `D` segments** to avoid redundant reversals.

This **greedy + reverse** approach allows us to efficiently construct the smallest number satisfying the conditions. 🚀