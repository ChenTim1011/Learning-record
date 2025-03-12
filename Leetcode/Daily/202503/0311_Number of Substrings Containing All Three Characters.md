[1358. Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/description/?envType=daily-question&envId=2025-03-11)

## **📌 Problem Statement**
We are given a string `s` consisting only of **'a', 'b', and 'c'**.  
We need to **count the number of substrings** that contain at least **one occurrence of all three characters ('a', 'b', and 'c')**.

---

## **🔹 Observations**
1. **Brute Force (O(n²)) is too slow**:
   - Checking all substrings would take **O(n²)**, which is too slow for **n = 50,000**.

2. **Sliding Window Approach (O(n))**:
   - Since we are looking for substrings, a **two-pointer sliding window** is efficient.
   - Expand `right` to include new characters.
   - Move `left` forward when all three characters are present.
   - **Each valid position of `left` contributes to multiple valid substrings**.

---

## **🔹 Approach**
1. **Use a hash map (or array) to count occurrences of 'a', 'b', 'c'**.
2. **Expand `right`**:
   - Include `s[right]` in the window.
3. **Move `left` forward** when all three characters exist:
   - Since **all substrings starting from `left` to the end of the string are valid**, count them.
   - `count += (n - right)`, where `n = s.length()`.
   - Reduce `s[left]` and move `left` forward.

---

## **💻 Code Implementation**
```cpp
class Solution {
public:
    int numberOfSubstrings(string s) {
        int count[3] = {0, 0, 0}; // Count of 'a', 'b', 'c'
        int left = 0, total = 0, n = s.length();

        for (int right = 0; right < n; right++) {
            count[s[right] - 'a']++;  // Include right character
            
            // If all characters are present, move left
            while (count[0] > 0 && count[1] > 0 && count[2] > 0) {
                total += (n - right);  // Every substring from [left, right] to [left, n-1] is valid
                count[s[left] - 'a']--; // Shrink window
                left++;
            }
        }
        
        return total;
    }
};
```

---

## **⏳ Complexity Analysis**
| **Operation**   | **Time Complexity** | **Space Complexity** |
|----------------|--------------------|--------------------|
| **Sliding Window Traversal** | **O(n)** | **O(1)** |

- **Time Complexity: O(n)**  
  - Each character is **processed at most twice** (once when expanding `right`, once when moving `left`).
- **Space Complexity: O(1)**  
  - We use a **fixed-size array** (`count[3]`).

---

## **✅ Summary**
| Approach | Time Complexity | Space Complexity | Notes |
|----------|---------------|----------------|----------------|
| **Brute Force (Nested Loops)** | **O(n²)** | **O(1)** | Too slow for large `n`. |
| **Sliding Window** | **O(n)** | **O(1)** | **Optimal and efficient** ✅ |

