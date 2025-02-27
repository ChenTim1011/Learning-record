[Length of Longest Fibonacci Subsequence](https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/description/)


## **Problem Statement**  
We are given a **strictly increasing** array `arr` of **positive integers**.  

A sequence **x₁, x₂, ..., xₙ** is **Fibonacci-like** if:
1. **It has at least 3 elements** (`n >= 3`).
2. **Each term is the sum of the two preceding terms:**  
   \[
   x_i + x_{i+1} = x_{i+2}
   \]
   for all \( i+2 \leq n \).

### **Goal:**  
Return the **length** of the **longest Fibonacci-like subsequence** in `arr`. If no valid sequence exists, return `0`.

A **subsequence** means we can **delete elements** from `arr` but cannot change the order.

---

## **Example Walkthrough**
### **Example 1**  
#### **Input:**  
```cpp
arr = [1,2,3,4,5,6,7,8]
```
#### **Valid Fibonacci-like Subsequences:**
- `[1,2,3,5,8]` (length = 5)
- `[2,3,5,8]` (length = 4)
- `[3,5,8]` (length = 3)
- `[1,2,3]` (length = 3)

#### **Output:**  
```cpp
5
```

---

### **Example 2**  
#### **Input:**  
```cpp
arr = [1,3,7,11,12,14,18]
```
#### **Valid Fibonacci-like Subsequences:**
- `[1,11,12]` (length = 3)
- `[3,11,14]` (length = 3)
- `[7,11,18]` (length = 3)

#### **Output:**  
```cpp
3
```

---

## **Approach 1: Brute Force (Exponential Time)**
### **Idea:**
- Try all possible subsequences of length **≥3**.
- Check if they follow the Fibonacci rule:  
  \[
  x_i + x_{i+1} = x_{i+2}
  \]
- Track the **maximum length** of such subsequences.

### **Complexity:**
- **Time Complexity:** `O(2^n)`, as we check all possible subsequences.
- **Space Complexity:** `O(n)`, for storing subsequences.

⚠ **This approach is too slow for `n ≤ 1000`**. We need a more efficient method.

---

## **Approach 2: Hash Set + Greedy Expansion (O(N² log M))**
### **Key Observations**
1. We can **avoid generating subsequences** explicitly.
2. Instead, we can **try pairing elements (`x`, `y`)** in `arr` and **expand** the sequence greedily.
3. To quickly check if a number exists, we use a **hash set**.

### **Algorithm**
1. **Store all elements in an unordered set** (`st`) for **O(1) lookups**.
2. **Iterate over all pairs (`arr[i]`, `arr[j]`)**:
   - Assume `arr[i]` and `arr[j]` are the **first two numbers** in a Fibonacci-like sequence.
   - Compute the **next number** (`arr[i] + arr[j]`).
   - If it exists in the set, **continue expanding** the sequence.
   - Track the **maximum length found**.

---

## **Optimized C++ Solution**
```cpp
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& arr) {
        unordered_set<int> st(arr.begin(), arr.end()); // Store all elements for quick lookup
        int maxLen = 0;

        // Try every pair (arr[i], arr[j]) as the first two elements of a Fibonacci sequence
        for (int i = 0; i < arr.size(); i++) {
            for (int j = i + 1; j < arr.size(); j++) {
                int x = arr[i], y = arr[j];
                int length = 2; // We already have two elements
                
                // Try to extend the sequence greedily
                while (st.count(x + y)) {
                    int z = x + y; // Next Fibonacci number
                    x = y;
                    y = z;
                    length++;
                }

                maxLen = max(maxLen, length);
            }
        }

        return maxLen > 2 ? maxLen : 0; // Return 0 if no valid sequence found
    }
};
```

---

## **Code Explanation**
### **1️⃣ Initialize Hash Set for Fast Lookup**
```cpp
unordered_set<int> st(arr.begin(), arr.end());
```
- Stores **all elements** of `arr` in an **unordered_set**.
- **Lookup operations** (`st.count(x)`) take **O(1) average time**.

---

### **2️⃣ Iterate Over All Pairs (arr[i], arr[j])**
```cpp
for (int i = 0; i < arr.size(); i++) {
    for (int j = i + 1; j < arr.size(); j++) {
```
- We assume `arr[i]` and `arr[j]` are the **first two numbers** in a Fibonacci sequence.

---

### **3️⃣ Expand the Sequence Greedily**
```cpp
int x = arr[i], y = arr[j];
int length = 2;  // Already counted arr[i] and arr[j]

while (st.count(x + y)) {
    int z = x + y; // Next Fibonacci number
    x = y;
    y = z;
    length++;
}
```
- Compute `x + y` to check if the next Fibonacci number exists.
- If yes, continue extending the sequence.

---

### **4️⃣ Track Maximum Length Found**
```cpp
maxLen = max(maxLen, length);
```
- Update `maxLen` whenever we find a longer Fibonacci-like subsequence.

---

### **5️⃣ Handle Edge Case: No Valid Sequence**
```cpp
return maxLen > 2 ? maxLen : 0;
```
- If `maxLen` remains ≤2, it means **no valid Fibonacci-like sequence** was found, so return `0`.

---

## **Complexity Analysis**
| Complexity | Analysis |
|------------|----------|
| **Time Complexity** | **O(N² log M)** – We iterate over all pairs `(arr[i], arr[j])`, and each Fibonacci sequence expansion is logarithmic in `M` (max value in `arr`). |
| **Space Complexity** | **O(N)** – The hash set stores all elements in `arr`. |

---

## **Why is this the Optimal Solution?**
- **Avoids generating all subsequences** (exponential time).
- **Uses hash set for O(1) lookups** instead of searching.
- **Greedy expansion ensures O(log M) operations per pair.**

💡 **This is the best possible solution for `n ≤ 1000` and values up to `10⁹`.** 🚀

---

## **🔹 Summary**
✅ **We use a hash set to efficiently check for Fibonacci-like sequences.**  
✅ **We iterate over all pairs (`arr[i]`, `arr[j]`) and greedily extend the sequence.**  
✅ **Final complexity is `O(N² log M)`, which is much faster than brute force.**  

