[Letter Tile Possibilities](https://leetcode.com/problems/letter-tile-possibilities/description/)

## **Problem Explanation**
We are given a string `tiles` consisting of characters (A-Z), and we need to **count all possible non-empty sequences** that can be formed using these tiles. Each tile can be used **at most once per sequence**, and since the tiles may contain duplicates, we need to ensure that we don’t count duplicate sequences multiple times.

---

## **Understanding the Example**
### **Example 1**
```cpp
Input: tiles = "AAB"
Output: 8
```
The possible sequences are:
```
"A", "B", "AA", "AB", "BA", "AAB", "ABA", "BAA"
```
This demonstrates that different sequences can be formed using repetition but should be counted uniquely.

---

## **Approach Breakdown**
### **1. Counting Frequency of Each Character**
The first step is to count how many times each letter appears in `tiles`. We use a **frequency array** `freq` where `freq[i]` represents the count of character `'A' + i'` in the input.

```cpp
vector<int> freq(26, 0);
for (char c: tiles)
    if (++freq[c - 'A'] == 1) sz++;
```
Example: `tiles = "AAB"`
- `freq = {2, 1, 0, 0, ..., 0}` (because `'A'` appears twice, `'B'` appears once).
- `sz = 2` (number of unique characters: `'A'` and `'B'`).

---

### **2. Sorting and Resizing the Frequency Array**
Since we only need to consider non-zero frequencies, we:
1. **Sort** `freq` in descending order (to prioritize the most frequent characters first, which can improve efficiency).
2. **Resize** `freq` to contain only the non-zero counts.

```cpp
sort(freq.begin(), freq.end(), greater<int>());
freq.resize(sz);
```
For `"AAB"`, this results in:
```
freq = {2, 1}
```

---

### **3. Generating Sequences Using Recursion**
Now, we use a **backtracking approach** to generate all valid sequences.

We call `Perm(n, freq, sz)` recursively to count how many unique permutations can be formed using **n tiles**.

#### **Recursive Function `Perm`**
```cpp
int Perm(int n, vector<int>& freq, int fz) {
    if (n == 1)  // Base case: Only one position left to fill
        return fz - count(freq.begin(), freq.end(), 0);
    
    int ans = 0;
    for (int i = 0; i < fz; i++) {
        if (freq[i] > 0) { // If the character is available
            freq[i]--;  // Use the character
            ans += Perm(n - 1, freq, fz); // Recursive call for remaining tiles
            freq[i]++;  // Backtrack (restore the count)
        }
    }
    return ans;
}
```
### **How `Perm` Works**
- If `n == 1`, we return the number of **nonzero frequency elements**, meaning how many different letters can still be chosen.
- Otherwise, we iterate through all available characters and recursively reduce `n`.
- **Backtracking** ensures that each possibility is explored without modifying data permanently.

---

### **4. Looping Over Different Sequence Lengths**
Since valid sequences can have lengths from `1` to `tiles.size()`, we sum the results for all possible lengths.

```cpp
int cnt = 0;
for (int len = 1; len <= tz; len++) 
    cnt += Perm(len, freq, sz);
```
This loop ensures that we count sequences of length **1, 2, ..., n**.

---

## **Step-by-Step Execution (Example: "AAB")**
### **Step 1: Preprocessing**
- `freq = {2, 1}` (A appears twice, B once)
- `sz = 2`
- Sorting does not change `freq` in this case.

### **Step 2: Counting Sequences of Length 1**
Calling `Perm(1, {2, 1}, 2)`:
- There are **two unique letters** (`A` and `B`), so the result is `2`.
- Running sum: `cnt = 2`.

### **Step 3: Counting Sequences of Length 2**
Calling `Perm(2, {2, 1}, 2)`:
- Choosing `A` first → `{1, 1}`
  - `Perm(1, {1, 1}, 2)` returns `2`
- Choosing `B` first → `{2, 0}`
  - `Perm(1, {2, 0}, 2)` returns `1`
- Total count for `n=2` is `3`.
- Running sum: `cnt = 2 + 3 = 5`.

### **Step 4: Counting Sequences of Length 3**
Calling `Perm(3, {2, 1}, 2)`:
- Choosing `A` first → `{1, 1}`
  - Choosing `A` again → `{0, 1}`
    - `Perm(1, {0, 1}, 2)` returns `1`
  - Choosing `B` → `{1, 0}`
    - `Perm(1, {1, 0}, 2)` returns `1`
- Choosing `B` first → `{2, 0}`
  - Choosing `A` → `{1, 0}`
    - `Perm(1, {1, 0}, 2)` returns `1`
- Total count for `n=3` is `3`.
- Running sum: `cnt = 5 + 3 = 8`.

---

## **Final Result**
`numTilePossibilities("AAB") = 8`

---

## **Time and Space Complexity**
### **Time Complexity**
- The number of unique sequences follows **factorial-like growth**, leading to an **O(n!)** complexity.
- Backtracking ensures that duplicate sequences are **not counted twice**.

### **Space Complexity**
- `freq` array has a max size of `O(26)`, which is **constant**.
- The recursive stack depth is **O(n)** in the worst case.

Thus, the space complexity is **O(n)**.

---

## **Summary**
1. **Use a frequency array** (`freq`) to count occurrences of each letter.
2. **Sort and resize `freq`** to retain only relevant values.
3. **Use a recursive function `Perm(n, freq, sz)`** to count all unique permutations using backtracking.
4. **Iterate over all sequence lengths** (`1` to `n`) and sum the results.

This method **efficiently handles duplicate letters** while ensuring all possible sequences are counted correctly. 🚀