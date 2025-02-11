[Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)

### Problem Description
The task is to determine if two strings, `s` and `t`, are **anagrams** of each other.  
An anagram is formed by rearranging the letters of a word to produce a new word, using the same characters the exact same number of times.

#### Examples:
1. Input: `s = "anagram"`, `t = "nagaram"`  
   Output: `true`  
   Explanation: Rearranging the letters of `"anagram"` can result in `"nagaram"`.

2. Input: `s = "rat"`, `t = "car"`  
   Output: `false`  
   Explanation: The letters in `"rat"` cannot be rearranged to form `"car"`.

#### Constraints:
- \( 1 \leq \text{s.length, t.length} \leq 50,000 \)
- Both strings consist of lowercase English letters.

---

### Solution Explanation

To check if two strings are anagrams:
1. Both strings must contain the same characters.
2. Each character must appear the same number of times in both strings.

The provided solution uses a **character frequency count** approach to solve the problem efficiently. This method works because:
- We only need to handle lowercase English letters, making it possible to use a fixed-size array of 26 elements to track character counts.

#### Steps:
1. Create an array `count[26]` initialized to 0. Each element of the array corresponds to a letter ('a' to 'z').
2. Traverse the first string `s` and increase the count for each character.
3. Traverse the second string `t` and decrease the count for each character.
4. Finally, check if all elements in the `count` array are 0. If any element is non-zero, it means the two strings are not anagrams.

---

### Annotated Code
```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        // Step 1: If the lengths of the strings are not the same, they can't be anagrams
        if (s.size() != t.size()) {
            return false;
        }

        // Step 2: Create a frequency array to track character counts
        // The array has 26 elements, one for each lowercase English letter
        int count[26] = {0};

        // Step 3: Traverse the first string `s` and increment the count for each character
        for (int i = 0; i < s.size(); i++) {
            count[s[i] - 'a']++; // Map character 'a'-'z' to indices 0-25
        }

        // Step 4: Traverse the second string `t` and decrement the count for each character
        for (int i = 0; i < t.size(); i++) {
            count[t[i] - 'a']--; // Decrease the count for the corresponding character
        }

        // Step 5: Check if all counts are zero
        // If any count is non-zero, it means `s` and `t` are not anagrams
        for (int i = 0; i < 26; i++) {
            if (count[i] != 0) {
                return false; // Unequal counts for some character
            }
        }

        // If we reach here, all counts are zero, so the strings are anagrams
        return true;
    }
};
```

---

### Explanation of Each Step in Detail

1. **Length Check**:
   - Before comparing characters, the function checks if the lengths of `s` and `t` are equal. If they aren't, the strings cannot be anagrams.

2. **Frequency Array**:
   - A fixed-size array `count[26]` is used to store the frequency of each letter in the strings.  
   - The index of each letter is determined by subtracting `'a'` from the ASCII value of the letter.  
     Example:  
     - `'a' - 'a' = 0` → Index 0 corresponds to `'a'`.
     - `'b' - 'a' = 1` → Index 1 corresponds to `'b'`.

3. **Counting Characters in `s`**:
   - For each character in `s`, increment its corresponding index in the `count` array.

4. **Counting Characters in `t`**:
   - For each character in `t`, decrement its corresponding index in the `count` array.  
   - If `t` is a valid anagram of `s`, all increments and decrements will balance out to 0.

5. **Final Validation**:
   - If all elements in the `count` array are 0, the strings are anagrams.
   - If any element is non-zero, it means one string has more or fewer occurrences of some character than the other, so they are not anagrams.

---

### Complexity Analysis

1. **Time Complexity**:  
   - \( O(n) \): The algorithm iterates over each string once, where \( n \) is the length of the strings.

2. **Space Complexity**:  
   - \( O(1) \): The frequency array `count[26]` is of fixed size and does not depend on the input size.

---

### Follow-Up: Unicode Characters
If the input strings contain Unicode characters, the fixed-size array `count[26]` is no longer sufficient. Instead:
1. Use a **hash map** (e.g., `unordered_map<char, int>` in C++) to store character frequencies.
2. The rest of the algorithm remains the same, but the space complexity will depend on the number of unique characters in the strings.

