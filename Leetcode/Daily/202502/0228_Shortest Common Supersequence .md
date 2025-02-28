[Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/description/?envType=daily-question&envId=2025-02-28)


## **Problem Statement**  
You are given **two strings** `str1` and `str2`.  

Return **the shortest string** that has **both** `str1` and `str2` as **subsequences**.  
If there are **multiple valid strings**, return **any** of them.  

A string `s` is a **subsequence** of string `t` if deleting **some** characters from `t` (possibly `0`) results in `s`.  

---

## **Example Walkthrough**
### **Example 1**  
#### **Input:**  
```cpp
str1 = "abac", str2 = "cab"
```
#### **Output:**  
```cpp
"cabac"
```
#### **Explanation:**  
- `"abac"` is a **subsequence** of `"cabac"` (remove the first `"c"`).
- `"cab"` is a **subsequence** of `"cabac"` (remove the last `"ac"`).
- `"cabac"` is the **shortest** string that satisfies these properties.

---

### **Example 2**  
#### **Input:**  
```cpp
str1 = "aaaaaaaa", str2 = "aaaaaaaa"
```
#### **Output:**  
```cpp
"aaaaaaaa"
```
#### **Explanation:**  
- Since both strings are **identical**, the shortest common supersequence is just `"aaaaaaaa"`.

---

## **Key Observations**
1. **The longest common subsequence (LCS) helps find the shortest supersequence.**
   - The more characters `str1` and `str2` **share**, the **shorter** the SCS.
2. **SCS Length Formula:**
   \[
   \text{SCS Length} = n + m - \text{LCS Length}
   \]
   - `n` = length of `str1`
   - `m` = length of `str2`
   - `LCS Length` = longest common subsequence length

---

## **Approach: Using Dynamic Programming (DP)**
### **Steps to Solve the Problem**
1. **Compute LCS** using DP.
2. **Backtrack through DP table** to construct the SCS.
   - Add characters **from both strings** while maintaining order.
   - Include **LCS characters only once**.
3. **Return the result**.

---

## **Optimized C++ Solution**
```cpp
class Solution {
public:
    string shortestCommonSupersequence(string str1, string str2) {
        int n1 = str1.size(), n2 = str2.size();
        vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 0));

        // Step 1: Compute LCS DP Table
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (str1[i - 1] == str2[j - 1])
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                else
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }

        // Step 2: Construct SCS using DP table
        int i = n1, j = n2;
        string result = "";
        
        while (i > 0 && j > 0) {
            if (str1[i - 1] == str2[j - 1]) {  
                result += str1[i - 1];  // Common character
                i--; j--;
            } 
            else if (dp[i - 1][j] > dp[i][j - 1]) {  
                result += str1[i - 1];  // Take from str1
                i--;
            } 
            else {  
                result += str2[j - 1];  // Take from str2
                j--;
            }
        }

        // Step 3: Append remaining characters (if any)
        while (i > 0) result += str1[--i];
        while (j > 0) result += str2[--j];

        // Step 4: Reverse the result string
        reverse(result.begin(), result.end());
        
        return result;
    }
};
```

---

## **Explanation of Code**
### **1️⃣ Compute LCS using Dynamic Programming**
We create a `dp` table where `dp[i][j]` stores the **LCS length** of `str1[0..i-1]` and `str2[0..j-1]`.  
```cpp
vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 0));

for (int i = 1; i <= n1; i++) {
    for (int j = 1; j <= n2; j++) {
        if (str1[i - 1] == str2[j - 1])
            dp[i][j] = 1 + dp[i - 1][j - 1];
        else
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
    }
}
```
- If characters **match**, extend the LCS (`1 + dp[i-1][j-1]`).
- Else, take the **maximum LCS** from either **excluding `str1[i-1]`** or **excluding `str2[j-1]`**.

---

### **2️⃣ Construct SCS using Backtracking**
- **If characters match**, add **only once** to `result` and **move diagonally**.
- **Else, take the larger LCS direction** (either `up` or `left`).
```cpp
int i = n1, j = n2;
string result = "";

while (i > 0 && j > 0) {
    if (str1[i - 1] == str2[j - 1]) {  
        result += str1[i - 1];  // Common character
        i--; j--;
    } 
    else if (dp[i - 1][j] > dp[i][j - 1]) {  
        result += str1[i - 1];  // Take from str1
        i--;
    } 
    else {  
        result += str2[j - 1];  // Take from str2
        j--;
    }
}
```

---

### **3️⃣ Add Remaining Characters**
- If there are **leftover** characters in either `str1` or `str2`, **append them**.
```cpp
while (i > 0) result += str1[--i];
while (j > 0) result += str2[--j];
```

---

### **4️⃣ Reverse the Result**
Since we built `result` **backwards**, we **reverse** it at the end.
```cpp
reverse(result.begin(), result.end());
return result;
```

---

## **Complexity Analysis**
| Complexity | Analysis |
|------------|----------|
| **Time Complexity** | **O(n * m) + O(n + m)** = **O(n * m)** |
| **Space Complexity** | **O(n * m)** (for `dp` table) |

---

## **Example Walkthrough**
### **Example: str1 = "brute", str2 = "groot"**
#### **1️⃣ LCS Table**
```
    g  r  o  o  t
 0  0  0  0  0  0
 b  0  0  0  0  0
 r  0  1  1  1  1
 u  0  1  1  1  1
 t  0  1  1  1  2
 e  0  1  1  1  2
```
#### **2️⃣ LCS = "rt"**
#### **3️⃣ Constructing SCS**
- Add `'e'` from `brute`
- Add `'t'` (from `LCS`)
- Add `'u'` from `brute`
- Add `'o'` from `groot`
- Add `'o'` from `groot`
- Add `'r'` (from `LCS`)
- Add `'b'` from `brute`
- Add `'g'` from `groot`

#### **4️⃣ Final Answer = `"gobruote"`**

---

## **Summary**
✅ **Find LCS using DP**  
✅ **Backtrack to construct SCS**  
✅ **Append remaining characters**  
✅ **Reverse and return the result**  

