[Minimum Recolors to Get K Consecutive Black Blocks](https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/description/?envType=daily-question&envId=2025-03-08)


## **📌 Problem Statement**
Given a **binary string** `blocks` of length `n`, where:
- `'W'` represents a **white block**.
- `'B'` represents a **black block**.

You are also given an **integer** `k`, which is the desired number of **consecutive black blocks**.

**In one operation**, you can **recolor** a white block (`'W'`) into a black block (`'B'`).

**Goal:** Find the **minimum number of operations** needed so that there is **at least one occurrence of `k` consecutive `'B'` blocks**.

---

## **🔹 Example**
### **Example 1**
```
Input:  blocks = "WBBWWBBWBW", k = 7
Output: 3
Explanation:
We can recolor the 0th, 3rd, and 4th blocks:
"WBBWWBBWBW" → "BBBBBBBWBW"
```

### **Example 2**
```
Input:  blocks = "WBWBBBW", k = 2
Output: 0
Explanation:
The substring "BB" already exists, so no changes are needed.
```

---

## **🚀 Approach**
### **🔑 Key Idea: Sliding Window**
- Use a **fixed-size sliding window** of length `k`.
- Count the number of **'W'** blocks within the window.
- **Slide the window** across the string, updating the count efficiently.
- The minimum number of **'W'** blocks in any window is the answer.

---

## **💻 Code (C++)**
```cpp
class Solution {
public:
    int minimumRecolors(string blocks, int k) {
        int minOperations = INT_MAX;
        int whiteCount = 0;

        // Count 'W' in the first window
        for (int i = 0; i < k; i++) {
            if (blocks[i] == 'W') whiteCount++;
        }
        minOperations = whiteCount;

        // Slide the window
        for (int i = k; i < blocks.size(); i++) {
            if (blocks[i - k] == 'W') whiteCount--; // Remove the outgoing element
            if (blocks[i] == 'W') whiteCount++; // Add the new element
            minOperations = min(minOperations, whiteCount);
        }

        return minOperations;
    }
};
```

---

## **⏳ Complexity Analysis**
| Approach        | Time Complexity | Space Complexity | Explanation |
|----------------|---------------|----------------|-------------|
| **Sliding Window** | **O(n)** | **O(1)** | We traverse `blocks` once with a fixed window |

---

## **✅ Summary**
| Method | Time Complexity | Space Complexity | Notes |
|--------|---------------|----------------|----------------|
| **Sliding Window** | **O(n)** | **O(1)** | Efficient approach for continuous segments |

This solution efficiently finds the **minimum recoloring operations** to get **k consecutive black blocks** using **Sliding Window**. 🚀
