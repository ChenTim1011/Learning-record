[Construct K Palindrome Strings
](https://leetcode.com/problems/construct-k-palindrome-strings/description/)

---

### **Problem Breakdown**
We need to determine if it's possible to construct exactly `k` palindrome strings using all the characters in a string `s`. Here are the key aspects:

1. **Palindrome Properties:**
   - A palindrome reads the same backward as forward.
   - In terms of character frequency:
     - Each character must appear an even number of times.
     - At most, **one character** can have an odd frequency (it becomes the center of the palindrome).

2. **Key Observations:**
   - For a string with a certain character distribution:
     - Characters with odd frequencies determine the **minimum number of palindromes** required.
     - Example: If there are 3 characters with odd frequencies, at least 3 palindromes are needed (each odd-frequency character acts as the center of one palindrome).

3. **Feasibility Conditions:**
   - **Condition 1:** If `k > s.length`, it’s impossible because we can’t have more palindromes than characters.
   - **Condition 2:** If the number of characters with odd frequencies (`oddCount`) is greater than `k`, it's also impossible because we can’t group the characters to form `k` palindromes.
   - Otherwise, it’s feasible.

---

### **Approach Explanation**
The solution uses a **sorting-based approach** to count character frequencies and check the conditions:

#### 1. **Sorting the String**
   - First, the string is sorted alphabetically (`sort(s.begin(), s.end())`).
   - This groups identical characters together, making it easy to count their frequencies in a single traversal.

#### 2. **Counting Odd Frequencies**
   - Traverse the sorted string and count the frequency of each character.
   - If a character’s frequency is odd, increment the `oddCount` counter.

#### 3. **Checking Feasibility**
   - If `s.length() < k`, return `false` immediately (Condition 1).
   - If `oddCount > k`, return `false` (Condition 2).
   - Otherwise, return `true`.

---

### **Step-by-Step Execution**

Let’s go through the solution with an example:

#### **Example 1: Input `s = "annabelle"`, `k = 2`**

- **Step 1: Sort the string**  
  After sorting, `s` becomes `"aabeellnn"`.

- **Step 2: Count frequencies and odd occurrences**  
  Traverse through the sorted string:
  - `a`: appears 2 times (even).
  - `b`: appears 1 time (odd → `oddCount = 1`).
  - `e`: appears 2 times (even).
  - `l`: appears 2 times (even).
  - `n`: appears 2 times (even).  
  Result: `oddCount = 1`.

- **Step 3: Check feasibility**
  - `oddCount (1) ≤ k (2)`: True.
  - `s.length (9) ≥ k (2)`: True.
  - Therefore, it’s feasible to construct 2 palindromes.

---

#### **Example 2: Input `s = "leetcode"`, `k = 3`**

- **Step 1: Sort the string**  
  After sorting, `s` becomes `"cdeeelot"`.

- **Step 2: Count frequencies and odd occurrences**  
  Traverse through the sorted string:
  - `c`: appears 1 time (odd → `oddCount = 1`).
  - `d`: appears 1 time (odd → `oddCount = 2`).
  - `e`: appears 3 times (odd → `oddCount = 3`).
  - `l`: appears 1 time (odd → `oddCount = 4`).
  - `o`: appears 1 time (odd → `oddCount = 5`).
  - `t`: appears 1 time (odd → `oddCount = 6`).  
  Result: `oddCount = 6`.

- **Step 3: Check feasibility**
  - `oddCount (6) > k (3)`: False.
  - It’s impossible to construct 3 palindromes.

---

### **Complexity Analysis**

1. **Time Complexity**
   - Sorting the string: \(O(n \log n)\), where \(n\) is the length of the string.
   - Traversing the sorted string to count frequencies: \(O(n)\).
   - Overall: \(O(n \log n)\).

2. **Space Complexity**
   - The algorithm uses no extra data structures other than variables (`oddCount`, etc.).
   - Space complexity: \(O(1)\).

---

### **Why This Works**
Sorting simplifies frequency grouping, making it straightforward to count odd occurrences in a single traversal. The conditions (`oddCount ≤ k` and `s.length ≥ k`) ensure that the characters can be rearranged into exactly `k` palindromes without leaving unused characters.

By focusing on odd frequencies, the solution guarantees correct palindromes while keeping the implementation efficient.