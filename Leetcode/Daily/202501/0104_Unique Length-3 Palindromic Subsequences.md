[Unique Length-3 Palindromic Subsequences](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/?envType=daily-question&envId=2025-01-04)

## **📌 Problem Statement**
Given a string `s`, return the **number of unique palindromes of length three** that are a **subsequence** of `s`.

🔹 **Definitions**:
- A **palindrome** is a string that reads the **same forwards and backwards**.
- A **subsequence** is a new string **formed by deleting** some characters (can be none) without changing the relative order of the remaining characters.

### **📝 Examples**
#### **Example 1**
**Input:**  
```cpp
s = "aabca"
```
**Valid Unique Palindromic Subsequences of Length 3**:
- `"aba"`
- `"aaa"`
- `"aca"`

**Output:**  
```cpp
3
```

#### **Example 2**
**Input:**  
```cpp
s = "adc"
```
**Output:**  
```cpp
0
```
**Explanation:** No palindromic subsequences exist.

#### **Example 3**
**Input:**  
```cpp
s = "bbcbaba"
```
**Valid Unique Palindromic Subsequences of Length 3**:
- `"bbb"`
- `"bcb"`
- `"bab"`
- `"aba"`

**Output:**  
```cpp
4
```

---

## **🔹 Intuition**
A **palindromic subsequence of length 3** follows the pattern:  
\[
(x, y, x)
\]
Where:
- **First (`x`) and last (`x`) characters must be the same**.
- **Middle (`y`) character can be any character**.

### **💡 Key Observations**
- To **form a valid palindrome**, find:
  1. The **leftmost occurrence** of a character `x`.
  2. The **rightmost occurrence** of the same character `x`.
  3. The **unique characters between them** as the middle character.

- **Count unique characters between leftmost and rightmost occurrences** for each character.

---

## **🔹 Optimized Approach**
### **1️⃣ Precompute First and Last Occurrences**
- **Find the first occurrence of each character** in `s`.
- **Find the last occurrence of each character** in `s`.

### **2️⃣ Count Unique Middle Characters**
- For each character `x` that appears at least **twice**:
  - Check **all characters** between its first and last occurrence.
  - Store **unique middle characters** in a `set`.
  - Count the number of distinct palindromes.

---

## **🔹 Efficient C++ Solution (O(n))**
```cpp
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        int n = s.length();
        int ans = 0;
        
        // Step 1: Find first and last occurrence of each character
        vector<int> first(26, -1);
        vector<int> last(26, -1);

        for (int i = 0; i < n; i++) {
            int index = s[i] - 'a';
            if (first[index] == -1) first[index] = i;
            last[index] = i;
        }

        // Step 2: Count unique characters between first and last occurrence
        for (int i = 0; i < 26; i++) {
            if (first[i] != -1 && last[i] - first[i] > 1) {
                unordered_set<char> middleChars;
                for (int j = first[i] + 1; j < last[i]; j++) {
                    middleChars.insert(s[j]);
                }
                ans += middleChars.size();
            }
        }

        return ans;
    }
};
```

---

## **🔹 Explanation**
### **Step 1: Precompute First and Last Occurrences**
```cpp
vector<int> first(26, -1);
vector<int> last(26, -1);
```
- We create two arrays `first` and `last` **(size 26 for lowercase letters)**.
- Traverse the string:
  - **Update `first[i]` when a character is seen for the first time**.
  - **Always update `last[i]` to track the rightmost occurrence**.

---

### **Step 2: Count Unique Middle Characters**
```cpp
for (int i = 0; i < 26; i++) {
    if (first[i] != -1 && last[i] - first[i] > 1) {
        unordered_set<char> middleChars;
        for (int j = first[i] + 1; j < last[i]; j++) {
            middleChars.insert(s[j]);
        }
        ans += middleChars.size();
    }
}
```
- If a character `x` appears **twice or more** (`first[i] != -1` and `last[i] - first[i] > 1`):
  - **Find all unique characters** between its first and last occurrence.
  - **Add the size of the set** (number of unique middle characters) to `ans`.

---

## **🔹 Complexity Analysis**
| Complexity | Explanation |
|------------|------------|
| **Time Complexity** | **O(n)** → We traverse `s` three times: once to find first/last occurrences, once to find unique middle characters. |
| **Space Complexity** | **O(1)** → Uses two fixed-size arrays (`first` and `last` of size `26`). |

---

## **🔹 Alternative Solution (Brute Force) - O(n²)**
We can check **all possible (x, y, x) subsequences**:
```cpp
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        unordered_set<string> uniquePalindromes;
        int n = s.size();

        for (int i = 0; i < n; i++) {
            for (int j = i + 2; j < n; j++) {
                if (s[i] == s[j]) {
                    for (int k = i + 1; k < j; k++) {
                        uniquePalindromes.insert({s[i], s[k], s[j]});
                    }
                }
            }
        }

        return uniquePalindromes.size();
    }
};
```

### **🔹 Complexity of Brute Force**
- **Time Complexity: O(n²)** → Nested loops iterating over `s`.
- **Space Complexity: O(n)** → Stores unique palindromic subsequences.

✅ **Brute force works for small inputs but is inefficient for `n = 10⁵`**.

---

## **🔹 Edge Cases Considered**
✅ **No palindromes exist** → `s = "abcde"` → **Output: 0**  
✅ **All characters are the same** → `s = "aaaa"` → **Output: 1 (`aaa`)**  
✅ **Characters appear out of order** → `s = "bbcbaba"` → **Handles multiple `x` correctly**  
✅ **Large input size `n = 10⁵`** → Efficient O(n) approach works.

---

## **🔹 Summary**
✅ **Efficient O(n) solution using first/last occurrences**  
✅ **Optimized space usage with fixed-size arrays**  
✅ **Handles edge cases correctly**  
✅ **Avoids unnecessary computations using hash sets**  

