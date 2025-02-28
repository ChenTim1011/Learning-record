[Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/description/?envType=study-plan-v2&envId=leetcode-75)


## **Problem Statement**  
You are given an **integer array** `nums` of length `n` and an integer `k`.  

Find a **contiguous subarray** of length exactly `k` that has the **maximum average value** and return this value.  

Any answer with a calculation error **less than 10⁻⁵** will be accepted.  

---

## **Example Walkthrough**
### **Example 1**  
#### **Input:**  
```cpp
nums = [1,12,-5,-6,50,3], k = 4
```
#### **Finding the Maximum Average:**
- We need to check **all subarrays of size `k = 4`**.
- The possible subarrays:
  1. `[1, 12, -5, -6] → (1 + 12 + (-5) + (-6)) / 4 = 2 / 4 = 0.5`
  2. `[12, -5, -6, 50] → (12 + (-5) + (-6) + 50) / 4 = 51 / 4 = 12.75`
  3. `[-5, -6, 50, 3] → (-5 + (-6) + 50 + 3) / 4 = 42 / 4 = 10.5`

✅ The maximum average is **`12.75`**.

#### **Output:**  
```cpp
12.75000
```

---

### **Example 2**  
#### **Input:**  
```cpp
nums = [5], k = 1
```
#### **Finding the Maximum Average:**
- The only subarray is `[5]`, and its average is **5.00000**.

#### **Output:**  
```cpp
5.00000
```

---

## **Approach 1: Brute Force (O(NK))**
### **Idea:**
- Try **all possible subarrays** of size `k`, compute their sum, and track the maximum.
- **Time Complexity:** **O(NK)** (Too slow for large `n`).

### **Algorithm:**
1. Loop over all possible **starting indices**.
2. Compute the sum of the **next `k` elements**.
3. Keep track of the **maximum sum**.
4. Return the **maximum average**.

---

## **Approach 2: Sliding Window (O(N), Optimal)**
### **Key Observations:**
1. Instead of recomputing the sum for each subarray, we can use a **sliding window**.
2. **Maintain a running sum of the current `k` elements**.
3. When moving the window to the right:
   - **Subtract the leftmost element** (exiting window).
   - **Add the new rightmost element** (entering window).
4. This allows us to update the sum in **O(1) time** instead of recomputing.

---

## **Optimized C++ Solution**
```cpp
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        double maxAvg = INT_MIN;
        double sum = 0.0;

        // Compute the sum of the first window of size k
        for (int i = 0; i < k; i++) {
            sum += nums[i];
        }

        maxAvg = sum / k;

        // Slide the window across the array
        for (int i = k; i < nums.size(); i++) {
            sum += nums[i] - nums[i - k];  // Add new element, remove old
            maxAvg = max(maxAvg, sum / k);
        }

        return maxAvg;
    }
};
```

---

## **Code Explanation**
### **1️⃣ Initialize the First Window**
```cpp
for (int i = 0; i < k; i++) {
    sum += nums[i];
}
maxAvg = sum / k;
```
- Compute the sum of the **first `k` elements**.
- Store the **average** in `maxAvg`.

---

### **2️⃣ Slide the Window Across the Array**
```cpp
for (int i = k; i < nums.size(); i++) {
    sum += nums[i] - nums[i - k];  // Add new, remove old
    maxAvg = max(maxAvg, sum / k);
}
```
- **Remove the leftmost element** (`nums[i - k]`).
- **Add the new element** (`nums[i]`).
- **Update the max average** if the new average is larger.

---

## **Complexity Analysis**
| Complexity | Analysis |
|------------|----------|
| **Time Complexity** | **O(N)** – We traverse the array once. |
| **Space Complexity** | **O(1)** – Only a few variables are used. |

---

## **Why is this the Optimal Solution?**
✅ **Avoids recomputing the sum from scratch** for each subarray (**O(NK) → O(N)**).  
✅ **Uses a fixed-size sliding window** with constant time updates.  
✅ **Handles large `n` efficiently** (up to **10⁵**).  

