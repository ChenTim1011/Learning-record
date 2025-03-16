[Minimum Time to Repair Cars](https://leetcode.com/problems/minimum-time-to-repair-cars/description/?envType=daily-question&envId=2025-03-16)

## **📌 Problem Statement**
We have an array `ranks` where each `ranks[i]` represents the **rank** of a mechanic.  
A mechanic with rank `r` can repair `n` cars in **`r * n²` minutes**.  
We need to find the **minimum time** required to repair all `cars` using multiple mechanics.

---

## **🎯 Key Observations**
- Higher rank **means slower repairs** because `r * n²` grows faster.
- The **mechanics work simultaneously**.
- The **number of cars each mechanic repairs is independent**, but the **total time is dictated by the slowest mechanic**.
- **Binary search on time** is the best approach.

---

## **🔹 Example Walkthrough**
### **Example 1**
#### **Input**
```cpp
ranks = [4,2,3,1], cars = 10
```
#### **Valid Assignments**
| Mechanic Rank | Cars Fixed | Time Used |
|--------------|-----------|------------|
| **4** | 2 | `4 × 2² = 16` |
| **2** | 2 | `2 × 2² = 8` |
| **3** | 2 | `3 × 2² = 12` |
| **1** | 4 | `1 × 4² = 16` |

#### **Output**
```cpp
16
```

---

### **Example 2**
#### **Input**
```cpp
ranks = [5,1,8], cars = 6
```
#### **Valid Assignments**
| Mechanic Rank | Cars Fixed | Time Used |
|--------------|-----------|------------|
| **5** | 1 | `5 × 1² = 5` |
| **1** | 4 | `1 × 4² = 16` |
| **8** | 1 | `8 × 1² = 8` |

#### **Output**
```cpp
16
```

---

## **🚀 Optimized Approach: Binary Search on Time**
### **🔹 Key Idea**
We binary search for **minimum required time `T`**, where:
- Each mechanic repairs as many cars as possible within `T`.
- If **total repaired cars ≥ required cars**, `T` is a valid candidate.
- Try **minimizing `T`** using binary search.

### **🔹 Steps**
1. **Initialize search bounds**  
   - **`left = 1`** (minimum time).
   - **`right = minRank * cars²`** (worst case, all cars by one mechanic).
  
2. **Binary Search on Time**
   - **Midpoint `T`**: check if `T` allows repairing at least `cars` cars.
   - If `T` is valid, **search left** (try smaller `T`).
   - Otherwise, **search right** (increase `T`).

---

## **📝 Implementation**
```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    // Helper function to check if `time` is enough to repair `cars`
    bool canRepair(vector<int>& ranks, int cars, long long time) {
        long long totalCars = 0;
        for (int r : ranks) {
            totalCars += sqrt(time / r); // Max cars this mechanic can repair in `time`
            if (totalCars >= cars) return true; // Early exit
        }
        return false;
    }

    long long repairCars(vector<int>& ranks, int cars) {
        long long left = 1, right = 1LL * (*min_element(ranks.begin(), ranks.end())) * cars * cars;
        
        while (left < right) {
            long long mid = left + (right - left) / 2;
            if (canRepair(ranks, cars, mid)) {
                right = mid; // Try to minimize time
            } else {
                left = mid + 1; // Increase time
            }
        }

        return left;
    }
};
```

---

## **⏳ Complexity Analysis**
| **Operation**     | **Time Complexity** | **Explanation** |
|------------------|------------------|----------------|
| **Binary Search on Time** | `O(log (minRank * cars²))` | Search range is large but logarithmic. |
| **Checking `canRepair()`** | `O(n log (minRank * cars²))` | Iterates over `n` mechanics per check. |
| **Total Complexity** | `O(n log (minRank * cars²))` | ✅ Efficient for constraints. |

🚀 **Optimized for `n = 10⁵` and `cars = 10⁶`!** ✅

---

## **✅ Summary**
- **Binary search on time `T`** efficiently finds the **minimum repair time**.
- **Each mechanic works independently**, so we sum their contributions.
- **Time Complexity:** `O(n log (minRank * cars²))` — **optimal** for large inputs.

