[House Robber IV](https://leetcode.com/problems/house-robber-iv/description/?envType=daily-question&envId=2025-03-15)

## **📌 Problem Statement**  
A robber wants to rob houses along a street, but **he cannot rob adjacent houses**.  
Each house has a certain amount of money stored in it, given in an array `nums`, where:  
- `nums[i]` represents the amount of money in the `i`-th house.  
- The robber must rob **at least `k` houses**.  
- The **capability of the robber** is the **maximum amount** of money he steals from a single house in his chosen sequence.  

### **Goal:**  
Find the **minimum possible capability** of the robber while still robbing at least `k` houses.  

---

## **🔹 Example Walkthrough**  

### **Example 1**  
#### **Input:**  
```cpp
nums = [2,3,5,9], k = 2
```
#### **Process:**  
- The robber needs to rob at least **2 houses**.  
- Possible ways to select houses (non-adjacent):  
  1. Rob **house 0** (`2`) and **house 2** (`5`) → capability = `max(2, 5) = 5`
  2. Rob **house 0** (`2`) and **house 3** (`9`) → capability = `max(2, 9) = 9`
  3. Rob **house 1** (`3`) and **house 3** (`9`) → capability = `max(3, 9) = 9`
- The **minimum** capability among these choices is `5`.

#### **Output:**  
```cpp
5
```

---

### **Example 2**  
#### **Input:**  
```cpp
nums = [2,7,9,3,1], k = 2
```
#### **Process:**  
- The robber needs to rob at least **2 houses**.  
- Possible ways to select houses (non-adjacent):  
  1. **House 0 (`2`) and house 4 (`1`)** → capability = `max(2, 1) = 2`
  2. **House 0 (`2`) and house 3 (`3`)** → capability = `max(2, 3) = 3`
  3. **House 1 (`7`) and house 3 (`3`)** → capability = `max(7, 3) = 7`
  4. **House 1 (`7`) and house 4 (`1`)** → capability = `max(7, 1) = 7`
  5. **House 2 (`9`) and house 4 (`1`)** → capability = `max(9, 1) = 9`
- The **minimum** capability among these choices is `2`.

#### **Output:**  
```cpp
2
```

---

## **🔹 Approach – Binary Search on Answer**
### **Key Observations**  
- The **robber’s capability** is the **largest single amount** he must steal.  
- If we fix a **capability `x`**, we want to check:  
  - **Can we rob at least `k` houses while ensuring each robbed house has at most `x` money?**  
- If **`x` is too small**, we won’t be able to rob enough houses.  
- If **`x` is too large**, we may be able to reduce it further.  

### **Binary Search Strategy**  
- **Define search space:**  
  - `left = min(nums)` → The smallest amount in any house (minimum capability).  
  - `right = max(nums)` → The largest amount in any house (maximum capability).  
- **Binary search on `mid = (left + right) / 2`** (candidate capability).  
- **Use a greedy approach** to check if we can rob `k` houses while ensuring `max(robbed_houses) <= mid`.  
- If possible, try a **smaller capability** (`right = mid - 1`), otherwise try a **larger capability** (`left = mid + 1`).  

---

## **🔹 C++ Solution**
```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minCapability(vector<int>& nums, int k) {
        int left = *min_element(nums.begin(), nums.end());
        int right = *max_element(nums.begin(), nums.end());
        int ans = right;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (canRob(nums, k, mid)) {
                ans = mid;  // Try for a smaller capability
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }

private:
    bool canRob(vector<int>& nums, int k, int maxCap) {
        int count = 0;
        int n = nums.size();

        for (int i = 0; i < n; i++) {
            if (nums[i] <= maxCap) {
                count++;
                i++;  // Skip next house (since adjacent houses cannot be robbed)
            }
        }
        return count >= k;
    }
};
```

---

## **🔹 Code Explanation**
### **Step 1: Binary Search Setup**
```cpp
int left = *min_element(nums.begin(), nums.end());
int right = *max_element(nums.begin(), nums.end());
int ans = right;
```
- **`left`** = smallest house value (`min(nums)`).  
- **`right`** = largest house value (`max(nums)`).  
- **We binary search between `left` and `right` to find the smallest valid capability.**  

---

### **Step 2: Binary Search on Capability**
```cpp
while (left <= right) {
    int mid = left + (right - left) / 2;

    if (canRob(nums, k, mid)) {  
        ans = mid;  // Update answer
        right = mid - 1;  // Try for a smaller capability
    } else {
        left = mid + 1;  // Increase capability
    }
}
```
- We try `mid = (left + right) / 2` as a **candidate capability**.  
- **If we can rob `k` houses** with `max(robbed_houses) ≤ mid`, we try for a **smaller** `mid`.  
- **If not**, increase `mid`.  

---

### **Step 3: Checking if We Can Rob `k` Houses (`canRob`)**
```cpp
bool canRob(vector<int>& nums, int k, int maxCap) {
    int count = 0;
    int n = nums.size();

    for (int i = 0; i < n; i++) {
        if (nums[i] <= maxCap) {
            count++;
            i++;  // Skip next house (since adjacent houses cannot be robbed)
        }
    }
    return count >= k;
}
```
- We greedily **rob houses** **≤ `maxCap`**, skipping adjacent houses.  
- If we can rob at least `k` houses, return `true`, otherwise `false`.  

---

## **🔹 Complexity Analysis**
| Complexity | Analysis |
|------------|----------|
| **Time Complexity** | **O(n log m)**, where `n` is the number of houses and `m = max(nums)`. We perform **O(log m)** binary search steps, and each step takes **O(n)** to check feasibility. |
| **Space Complexity** | **O(1)**, since we use only a few extra variables. |

---

## **🔹 Summary**
✅ **Binary search is used to find the smallest possible capability that allows robbing `k` houses.**  
✅ **The greedy approach efficiently checks feasibility for a given capability.**  
✅ **The solution runs in `O(n log m)`, which is optimal given the constraints (`n ≤ 10^5`).**  

