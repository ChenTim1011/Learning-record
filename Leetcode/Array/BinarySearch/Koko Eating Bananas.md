[Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/description/)

## **📌 Problem Breakdown**
We have `n` piles of bananas where `piles[i]` represents the number of bananas in the `i`-th pile.  

Koko eats at a speed of **k bananas per hour**.  
- Each hour, she picks **one pile** and eats `k` bananas.  
- If a pile has fewer than `k` bananas, she finishes the pile and waits for the hour to end.  
- She needs to finish all bananas in **at most** `h` hours.

### **🎯 Goal**
Find the **minimum integer** value of `k` such that **Koko can eat all the bananas in `h` hours**.

---

## **🔹 Example Walkthrough**
### **Example 1**
#### **Input**
```cpp
piles = [3,6,7,11]
h = 8
```
#### **Valid Speeds**
| Eating Speed `k` | Total Hours Taken |
|------------------|------------------|
| **k = 4** ✅ | **8 hours** (valid) |
| k = 3 ❌ | 9 hours (too slow) |
| k = 5 ✅ | 7 hours (valid) |
| ... | ... |

#### **Output**
```cpp
4
```
---

### **Example 2**
#### **Input**
```cpp
piles = [30,11,23,4,20]
h = 5
```
#### **Valid Speeds**
| Eating Speed `k` | Total Hours Taken |
|------------------|------------------|
| k = 30 ✅ | **5 hours** (valid) |
| k = 29 ❌ | 6+ hours (too slow) |
| k = 40 ✅ | 5 hours (valid) |

#### **Output**
```cpp
30
```
---

### **Example 3**
#### **Input**
```cpp
piles = [30,11,23,4,20]
h = 6
```
#### **Valid Speeds**
| Eating Speed `k` | Total Hours Taken |
|------------------|------------------|
| **k = 23** ✅ | **6 hours** (valid) |
| k = 22 ❌ | 7+ hours (too slow) |
| k = 25 ✅ | 6 hours (valid) |

#### **Output**
```cpp
23
```
---

## **🚀 Optimized Approach: Binary Search**
### **🔹 Key Observations**
1. **Minimum possible `k`**  
   - `k` cannot be less than `1` (smallest speed).  

2. **Maximum possible `k`**  
   - `k` cannot be more than `max(piles)`, because if `k >= max(piles)`, Koko finishes each pile in **one** hour.

3. **Binary Search on `k`**  
   - Try mid-point `k = (left + right) / 2`  
   - Check if Koko **can finish eating** at this speed within `h` hours.  
   - If she can, try a **lower `k`** to minimize it.  
   - If she **can't**, try a **higher `k`**.

---

## **📝 Implementation**
```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    // Helper function to check if Koko can eat all bananas within `h` hours at `speed` k
    bool canFinish(vector<int>& piles, int h, int k) {
        int hoursNeeded = 0;
        for (int bananas : piles) {
            hoursNeeded += (bananas + k - 1) / k;  // Equivalent to ceil(bananas / k)
        }
        return hoursNeeded <= h;
    }

    int minEatingSpeed(vector<int>& piles, int h) {
        int left = 1, right = *max_element(piles.begin(), piles.end());
        int answer = right;

        while (left <= right) {
            int mid = left + (right - left) / 2;  // Try mid as eating speed
            if (canFinish(piles, h, mid)) {
                answer = mid;  // Update the best possible k
                right = mid - 1;  // Try a smaller k
            } else {
                left = mid + 1;  // Increase k to eat faster
            }
        }
        
        return answer;
    }
};
```

---

## **⏳ Complexity Analysis**
| **Operation**     | **Time Complexity** | **Explanation** |
|------------------|------------------|----------------|
| **Binary Search on `k`** | `O(log M)` | Searching from `1` to `max(piles)`. |
| **Checking `canFinish()`** | `O(n)` | Iterates through `piles` to calculate total hours. |
| **Total Complexity** | `O(n log M)` | Efficient for large inputs. |

Where:
- `n` = number of piles (`≤ 10^4`)
- `M` = max number of bananas in a pile (`≤ 10^9`)

**🚀 Optimized for large constraints!** ✅

---

## **✅ Summary**
- **Binary Search on `k`** efficiently finds the minimum speed.
- **`canFinish()` helper** calculates hours required for a given `k`.
- **Time Complexity:** `O(n log M)`, making it **optimal** for large constraints.
