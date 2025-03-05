[Count Total Number of Colored Cells](https://leetcode.com/problems/count-total-number-of-colored-cells/description/)

## **📌 Problem Statement**
We have an **infinite** 2D grid where each cell is initially **uncolored**. Given an integer `n`, we perform the following operations:

1. **Minute 1** → Color one arbitrary cell **blue**.
2. **Each following minute** → Color every **uncolored** cell that is **adjacent** to any blue cell.

We need to return the **total number of colored cells** at the end of `n` minutes.

---

## **🔎 Observing the Growth Pattern**
Let's visualize how the grid expands:

- **n = 1** → `1` (Only one cell)
- **n = 2** → `5` (Expands into a **cross** shape)
- **n = 3** → `13`
- **n = 4** → `25`

### **Pattern Formation**
Looking at the series:

```
1, 5, 13, 25, ...
```
We can observe that the growth pattern forms a **diamond shape**.

### **Mathematical Formula**
The number of **colored** cells at minute `n` follows the formula:

\[
\text{Total Cells} = 2n(n-1) + 1
\]

This accounts for:
- The **central** cell (`+1`).
- The **diamond growth** (`2n(n-1)`).

---

## **🚀 Optimized C++ Solution**
```cpp
class Solution {
public:
    long long coloredCells(int n) {
        return 2LL * n * (n - 1) + 1;
    }
};
```

---

## **💡 Complexity Analysis**
| Complexity  | Explanation |  
|------------|------------|  
| **Time Complexity** | **O(1)** → Direct formula calculation. |  
| **Space Complexity** | **O(1)** → Only uses a single variable. |

---

## **✅ Edge Cases Considered**
1. **Smallest Case** → `n = 1` (should return `1`).
2. **Larger Cases** → `n = 10^6` (should handle large numbers correctly).
3. **Formula Verification** → Tested against manual calculations.

---

## **🔹 Summary**
✅ **Optimized O(1) formula-based solution.**  
✅ **Uses pattern recognition for fast calculation.**  
✅ **Handles large values without overflow (`long long`).**