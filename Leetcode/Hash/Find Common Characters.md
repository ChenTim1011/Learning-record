[Find Common Characters](https://leetcode.com/problems/find-common-characters/description/)


### Explanation of the Code

This solution determines all the characters that appear in every string in the input `words` array, including duplicates. The idea is to use an array to keep track of the minimum frequency of each character across all strings.

---

#### Steps:

1. **Initialize the frequency array (`maxcount`)**:  
   The `maxcount` array stores the frequency of each character ('a' to 'z') in the first string of the array. This will serve as the baseline for comparison.

2. **Iterate through the rest of the strings**:  
   For each string in the `words` array, calculate the frequency of characters in that string using a temporary array (`count`). Update `maxcount` to reflect the minimum frequency of each character across all strings processed so far.

3. **Extract common characters**:  
   For each character, check how many times it appears across all strings (based on `maxcount`). If it appears one or more times, add it to the result for each occurrence.

4. **Return the result**:  
   The `result` vector contains the common characters (including duplicates) as strings, which is then returned.

---

### Annotated Code with Comments

```cpp
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        vector<string> result;  // To store the common characters
        int maxcount[26] = {0}; // Array to store the minimum frequency of each character ('a' to 'z')

        // Initialize maxcount with frequencies from the first word
        for (int j = 0; j < words[0].size(); j++) {
            maxcount[words[0][j] - 'a']++; // Increment the count for the character
        }

        // Process the remaining words to find common characters
        for (int i = 1; i < words.size(); i++) {
            int count[26] = {0}; // Temporary array to store character frequencies for the current word

            // Count the frequency of each character in the current word
            for (int j = 0; j < words[i].size(); j++) {
                count[words[i][j] - 'a']++;
            }

            // Update maxcount to keep the minimum frequency of each character
            for (int k = 0; k < 26; k++) {
                maxcount[k] = min(maxcount[k], count[k]);
            }
        }

        // Extract common characters from maxcount
        for (int i = 0; i < 26; i++) {
            while (maxcount[i] > 0) { // If the character appears in all words
                result.push_back(string(1, 'a' + i)); // Convert the character to a string and add to result
                maxcount[i]--; // Decrement the count
            }
        }

        return result; // Return the list of common characters
    }
};
```

---

### Explanation of Key Parts

1. **Using `maxcount` to Track Minimum Frequency**:  
   This ensures that only characters that appear in every word (up to the minimum count) are considered.  
   For example:
   - If `words = ["bella", "label", "roller"]`, then:
     - `'e'` appears 1 time in all words → `'e'` is added 1 time.
     - `'l'` appears at least 2 times in all words → `'l'` is added 2 times.

2. **Temporary Frequency Array (`count`)**:  
   Each string has its own frequency count, which is used to update `maxcount`. This prevents the characters from being mistakenly carried over between different strings.

3. **Converting Characters to Strings**:  
   The function `string(1, 'a' + i)` is used to convert a single character (`'a' + i`) into a string, which is required since the `result` vector stores strings.

---

### Example Walkthrough

#### Input:
`words = ["bella", "label", "roller"]`

1. **Step 1: Initialize `maxcount` using the first word:**
   - `words[0] = "bella"` → `maxcount = [1, 1, 0, 0, 1, ..., 0]`  
     (Counts: `'b' = 1, 'e' = 1, 'l' = 2, 'a' = 1`)

2. **Step 2: Process the second word (`"label"`):**
   - `count for "label" = [1, 1, 0, 0, 1, ..., 0]`  
   - Update `maxcount = min(maxcount, count)` → `maxcount = [1, 1, 0, 0, 1, ..., 0]`.

3. **Step 3: Process the third word (`"roller"`):**
   - `count for "roller" = [0, 0, 0, 0, 1, ..., 0]`  
   - Update `maxcount = min(maxcount, count)` → `maxcount = [0, 0, 0, 0, 1, ..., 0]`.

4. **Step 4: Extract common characters:**
   - Add `'e'` and `'l'` to the result as they appear at least once in all words.

#### Output:
`["e", "l", "l"]`

---

### Complexity Analysis

1. **Time Complexity**:
   - **Character Frequency Counting**: `O(n * m)`, where `n` is the number of words and `m` is the average length of a word.
   - **Result Construction**: `O(26)` (constant, as the alphabet size is fixed).
   - Total: `O(n * m)`.

2. **Space Complexity**:
   - **`maxcount` and `count` Arrays**: `O(26)` each.
   - **Result Vector**: Depends on the number of common characters.
   - Total: `O(26)` or `O(1)` (constant space for character frequency arrays).

---

This code is efficient for the constraints and provides the correct output for any valid input.