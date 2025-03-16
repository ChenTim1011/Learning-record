[Maximum Candies Allocated to K Children](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/description/?envType=daily-question&envId=2025-03-14)

## **📌 Problem Statement**  
We are given:  
- An array `candies`, where `candies[i]` represents the number of candies in the `i`-th pile.  
- An integer `k`, which represents the number of children.  

### **Rules:**  
- We can divide each pile into **smaller sub-piles** (i.e., split candies), but **we cannot merge** two piles together.  
- Each child **must receive the same number of candies**.  
- A child **can take candies from only one pile** (i.e., no combining candies from multiple piles).  
- Some piles may **remain unused**.  
- Our goal is to **maximize the number of candies each child gets** while distributing candies **to exactly `k` children**.  

### **Output:**  
Return the **maximum number of candies** that can be given **to each child** under these constraints.  

---

## **🔹 Example Walkthrough**  

### **Example 1**  
#### **Input:**  
```cpp
candies = [5,8,6], k = 3
```
#### **Process:**  
1. We have `3` children.  
2. We want to **maximize** the candies each child gets.  
3. We can split some piles:  
   - Split `candies[1] = 8` into **5 + 3**.  
   - Split `candies[2] = 6` into **5 + 1**.  
4. Now we have these piles: `[5, 5, 3, 5, 1]`.  
5. We can distribute **three piles of 5** to `k = 3` children.  

#### **Output:**  
```cpp
5
```
Each child receives **5** candies.

---

### **Example 2**  
#### **Input:**  
```cpp
candies = [2,5], k = 11
```
#### **Process:**  
1. We only have `7` candies in total (`2 + 5 = 7`).  
2. There are `11` children.  
3. **It's impossible** to give each child at least `1` candy.  

#### **Output:**  
```cpp
0
```
Each child receives **0** candies because we cannot distribute them equally.

---

## **🔹 Approach – Binary Search on Answer**
### **Key Observations**  
- The **maximum possible number of candies per child** is `max(candies)`, meaning no child can get more than the largest pile.  
- The **minimum possible number of candies per child** is `0` (if we can’t distribute candies to all `k` children).  
- **We can use binary search** to efficiently determine the maximum candies per child.  

### **Binary Search Strategy**  
- **Define the search range**:  
  - **Left bound (`low`)** = 1 (minimum candy per child).  
  - **Right bound (`high`)** = `max(candies)` (the largest candy pile).  
- **Binary search on `mid` (candies per child):**  
  - Check if we can distribute `mid` candies to `k` children.  
  - If **possible**, increase `mid` (try giving more candies).  
  - If **not possible**, decrease `mid`.  

---

## **🔹 C++ Solution**
```cpp
class Solution {
public:
    bool canDistribute(vector<int>& candies, long long k, int mid) {
        long long count = 0;  
        for (int c : candies) {
            count += c / mid;  // Count how many children we can satisfy with `mid` candies each.
        }
        return count >= k;  // Can we satisfy at least `k` children?
    }

    int maxCandies(vector<int>& candies, long long k) {
        if (accumulate(candies.begin(), candies.end(), 0LL) < k) return 0;  
        // If total candies < k, it's impossible to give every child at least 1.

        int low = 1, high = *max_element(candies.begin(), candies.end());
        int ans = 0;  

        while (low <= high) {
            int mid = low + (high - low) / 2;  // Avoid overflow
            if (canDistribute(candies, k, mid)) {  
                ans = mid;  // Update answer
                low = mid + 1;  // Try for a larger mid
            } else {
                high = mid - 1;  // Reduce `mid`
            }
        }
        return ans;
    }
};
```

---

## **🔹 Code Explanation**
### **Step 1: Helper Function – `canDistribute`**
```cpp
bool canDistribute(vector<int>& candies, long long k, int mid) {
    long long count = 0;  
    for (int c : candies) {
        count += c / mid;  // Count how many children we can satisfy with `mid` candies each.
    }
    return count >= k;  // Can we satisfy at least `k` children?
}
```
- Given `mid`, this function checks whether we can give `mid` candies to `k` children.  
- It iterates through `candies` and counts how many complete `mid`-sized portions can be made.  
- If `count >= k`, it means we **can** distribute at least `k` children.

---

### **Step 2: Binary Search in `maxCandies`**
```cpp
if (accumulate(candies.begin(), candies.end(), 0LL) < k) return 0;  
```
- If the **total number of candies** is less than `k`, we **immediately return `0`**, since it's impossible to give at least `1` candy to each child.

---

### **Step 3: Define Search Range**
```cpp
int low = 1, high = *max_element(candies.begin(), candies.end());
```
- The **minimum possible** candies per child is `1`.  
- The **maximum possible** is `max(candies)`, since a child **cannot get more than the largest pile**.

---

### **Step 4: Binary Search**
```cpp
while (low <= high) {
    int mid = low + (high - low) / 2;  // Avoid overflow
    if (canDistribute(candies, k, mid)) {  
        ans = mid;  // Update answer
        low = mid + 1;  // Try for a larger mid
    } else {
        high = mid - 1;  // Reduce `mid`
    }
}
```
1. **Calculate `mid`** (candies per child).  
2. **Check if it's possible to distribute `mid` candies per child** using `canDistribute()`.  
3. If **possible**, update `ans` and try for **larger `mid`** (increase `low`).  
4. If **not possible**, reduce `mid` (decrease `high`).  

---

## **🔹 Complexity Analysis**
| Complexity | Analysis |
|------------|----------|
| **Time Complexity** | **O(n log m)**, where `n` is the number of elements in `candies` and `m` is `max(candies)`. This is because we perform **O(log m)** binary search steps, and each step takes **O(n)** to check feasibility. |
| **Space Complexity** | **O(1)**, since we use only a few extra variables. |

---

## **🔹 Summary**
✅ **This problem is a classic "search on answer" problem that can be solved efficiently using binary search.**  
- **Key Idea**: Try different values of `mid` (candies per child) and **use binary search** to find the maximum feasible value.  
- **Binary search helps us efficiently find the best answer instead of brute-force checking every possibility.**  
- **O(n log m) complexity makes it feasible for large inputs.**  

