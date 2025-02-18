[Construct the Lexicographically Largest Valid Sequence](https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/description/)

### **Problem Breakdown**  
Given an integer \( n \), we need to construct a sequence of length \( 2n - 1 \) that satisfies the following conditions:  

1️⃣ The number **1 appears exactly once**.  
2️⃣ Each number from **2 to \( n \) appears twice**.  
3️⃣ The two occurrences of a number \( i \) must be placed **exactly \( i \) indices apart**.  
4️⃣ Among all possible valid sequences, we must return the **lexicographically largest one**.  

### **Example**
#### **Example 1**  
```cpp
Input: n = 3
Output: [3,1,2,3,2]
```
Explanation:  
There are two possible valid sequences:  
- **[3,1,2,3,2]**  
- **[2,3,2,1,3]**  
Since 3 appears earlier in **[3,1,2,3,2]**, it is lexicographically larger.

#### **Example 2**  
```cpp
Input: n = 5
Output: [5,3,1,4,3,5,2,4,2]
```

---

## **📝 Approach**
We will use **Backtracking** to try all valid placements for each number from \( n \) to 1.

### **🔹 Step 1: Use Backtracking**
Backtracking allows us to explore different placements of numbers, and **undo choices if they lead to an invalid sequence**.  

### **🔹 Step 2: Track Used Numbers**
We maintain a `used` array to **avoid placing the same number more than twice**.

### **🔹 Step 3: Prioritize Larger Numbers**
To ensure the **lexicographically largest** sequence, we **place larger numbers first**.

---

## **📌 C++ Code**
```cpp
class Solution {
public:
    vector<int> constructDistancedSequence(int n) {
        vector<int> result(2 * n - 1, 0); // Create a sequence of size (2n-1) initialized with 0s
        vector<bool> used(n + 1, false); // To track used numbers
        backtrack(result, used, n, 0); // Start backtracking from index 0
        return result;
    }

private:
    bool backtrack(vector<int>& result, vector<bool>& used, int n, int index) {
        // Move to the next available empty position
        while (index < result.size() && result[index] != 0) {
            index++;
        }
        // If we have successfully placed all numbers, return true
        if (index == result.size()) {
            return true;
        }

        // Try placing numbers from n to 1 (larger numbers first)
        for (int i = n; i >= 1; i--) {
            if (used[i]) continue; // Skip if number is already used

            if (i == 1) { // Number 1 only appears once
                result[index] = 1;
                used[1] = true;
                if (backtrack(result, used, n, index + 1)) return true; // Continue recursion
                result[index] = 0; // Undo choice
                used[1] = false;
            } 
            else if (index + i < result.size() && result[index + i] == 0) { // Check if we can place i at index and index + i 
                result[index] = i;
                result[index + i] = i;
                used[i] = true;
                if (backtrack(result, used, n, index + 1)) return true;
                result[index] = 0; // Undo placement
                result[index + i] = 0;
                used[i] = false;
            }
        }
        return false; // If no valid placement, return false
    }
};
```

---

## **📌 Explanation of the Code**
### **1️⃣ Construct an Empty Sequence**
```cpp
vector<int> result(2 * n - 1, 0);
```
- We create an array of size \( 2n - 1 \), initialized with `0`.  
- `0` represents empty spaces where numbers will be placed.

### **2️⃣ Keep Track of Used Numbers**
```cpp
vector<bool> used(n + 1, false);
```
- `used[i]` helps to **avoid placing the same number more than twice**.

### **3️⃣ Use Backtracking to Place Numbers**
```cpp
bool backtrack(vector<int>& result, vector<bool>& used, int n, int index)
```
- This function **tries to place each number from \( n \) to 1** in the sequence.

### **4️⃣ Find the First Empty Position**
```cpp
while (index < result.size() && result[index] != 0) {
    index++;
}
```
- Skip already filled positions.

### **5️⃣ Check If We've Successfully Filled the Sequence**
```cpp
if (index == result.size()) {
    return true;
}
```
- If `index` reaches the end, **we have a valid sequence**.

### **6️⃣ Place the Largest Possible Number First**
```cpp
for (int i = n; i >= 1; i--) {
```
- We try numbers **from \( n \) down to 1**, so that we get the **largest lexicographical order**.

### **7️⃣ Special Case: Placing 1**
```cpp
if (i == 1) {
    result[index] = 1;
    used[1] = true;
    if (backtrack(result, used, n, index + 1)) return true;
    result[index] = 0;
    used[1] = false;
}
```
- Since `1` appears **only once**, we place it and move forward.

### **8️⃣ Placing Larger Numbers**
```cpp
else if (index + i < result.size() && result[index + i] == 0) {
```
- We check if we can place **both occurrences of \( i \)** at positions `index` and `index + i`.

```cpp
result[index] = i;
result[index + i] = i;
used[i] = true;
if (backtrack(result, used, n, index + 1)) return true;
result[index] = 0;
result[index + i] = 0;
used[i] = false;
```
- If the placement **fails later**, we **undo** it and try a different number.

---

## **🔎 Complexity Analysis**
### **Time Complexity:**  
- In the **worst case**, we try all possible sequences.
- The complexity is approximately **\( O(2^n) \)**, but pruning reduces it.

### **Space Complexity:**  
- **\( O(n) \)** for the `used` array and recursion stack.

---

## **🔹 Summary**
✅ **Backtracking:** Try numbers, undo when necessary.  
✅ **Track Used Numbers:** Avoid duplicate placements.  
✅ **Lexicographical Order:** Start with **largest number**.  
✅ **Validating Conditions:** Ensure distance constraints hold.  

---

## **🚀 Final Thoughts**
This backtracking approach efficiently constructs the **lexicographically largest valid sequence** while ensuring all constraints are met. Although the problem has exponential complexity in the worst case, pruning significantly reduces unnecessary computations, making it feasible for \( n \leq 20 \). 🚀