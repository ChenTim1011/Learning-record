[Find Resultant Array After Removing Anagrams](https://leetcode.com/problems/find-resultant-array-after-removing-anagrams/description/)


### Problem Overview:
You are given a list of words, and the task is to remove consecutive anagrams from the list. An **anagram** is a word that can be formed by rearranging the letters of another word using all the original letters exactly once. The problem is asking you to repeatedly remove adjacent words that are anagrams until no more anagrams exist next to each other.

### Steps to Approach the Solution:

1. **Understanding Anagrams:**
   Two words are anagrams if they contain the exact same characters with the same frequencies, but potentially in a different order. For example, "abba" and "baba" are anagrams because they both have the characters 'a' and 'b' in equal quantities.

2. **Conditions for Removal:**
   If two consecutive words in the list are anagrams of each other, the second word should be removed from the list. We continue this process until there are no adjacent anagrams left.

3. **Efficient Anagram Check:**
   The most common method to check if two strings are anagrams is by comparing their sorted versions. However, in this solution, we use a more efficient approach by counting the frequency of each character in the words using a frequency array. This is faster than sorting and is ideal for this problem because the words have a fixed length limit.

### Plan:
1. **isAnagram Function:**
   - We create a helper function `isAnagram` that checks if two words are anagrams by comparing the frequency of characters in both words.
   - We use an array `count[26]` to keep track of the character frequencies for 'a' to 'z'. If the frequencies match after processing both words, the words are anagrams.

2. **removeAnagrams Function:**
   - We initialize a result vector to store the non-anagram words.
   - We always add the first word to the result.
   - For each subsequent word, we check if it is an anagram of the previous word. If it is, we skip adding it; if not, we add it to the result.

### Code Implementation:

```cpp
class Solution {
public:
    // Function to check if two words are anagrams
    bool isAnagram(string word1, string word2) {
        int count[26] = {0};  // Array to store frequency of each letter
        
        // If the lengths of word1 and word2 are not the same, they can't be anagrams
        if (word1.size() != word2.size()) {
            return false;
        }
        
        // Count frequency of characters in word1
        for (int i = 0; i < word1.size(); i++) {
            count[word1[i] - 'a']++;  // Map 'a' to index 0, 'b' to index 1, ..., 'z' to index 25
        }
        
        // Count frequency of characters in word2 (decrement the count)
        for (int i = 0; i < word2.size(); i++) {
            count[word2[i] - 'a']--;  // Decrement for each character in word2
        }
        
        // If any count is not zero, the words are not anagrams
        for (int i = 0; i < 26; i++) {
            if (count[i] != 0) {
                return false;  // Words are not anagrams if any frequency doesn't match
            }
        }
        
        return true;  // Words are anagrams if all counts are zero
    }

    // Function to remove anagrams from a list of words
    vector<string> removeAnagrams(vector<string>& words) {
        vector<string> result;
        result.push_back(words[0]);  // Always include the first word
        
        for (int i = 1; i < words.size(); i++) {
            // Check if the current word is an anagram of the previous one
            if (isAnagram(words[i], words[i - 1])) {
                continue;  // Skip the current word if it's an anagram
            }
            result.push_back(words[i]);  // Add the current word to the result
        }
        
        return result;
    }
};
```

### Detailed Explanation of Code:

1. **isAnagram Function:**
   - **Input:** Two words: `word1` and `word2`.
   - **Approach:**
     - First, we check if the lengths of the two words are the same. If not, they cannot be anagrams, and we return `false`.
     - We then initialize a frequency array `count[26]` to store the frequency of each letter (from 'a' to 'z'). We use the formula `word[i] - 'a'` to map each character to an index between 0 and 25.
     - We loop through both `word1` and `word2`. For each character in `word1`, we increment the count for that letter, and for each character in `word2`, we decrement the count.
     - After processing both words, if any value in the `count` array is non-zero, the words are not anagrams, so we return `false`. Otherwise, the words are anagrams, and we return `true`.

2. **removeAnagrams Function:**
   - **Input:** A vector of strings `words`.
   - **Approach:**
     - We initialize an empty result vector and add the first word from the input list to it, as there is no previous word to compare it with.
     - We loop through the remaining words in the list, comparing each word with the previous word. If the two words are anagrams (checked using `isAnagram`), we skip the current word. If they are not anagrams, we add the current word to the result.
     - Finally, we return the result vector.

### Example Walkthrough:

#### Example 1:
```cpp
Input: words = ["abba", "baba", "bbaa", "cd", "cd"]
```

1. Initially, `result = ["abba"]`.
2. Compare "baba" with "abba" — they are anagrams, so we skip "baba".
3. Compare "bbaa" with "baba" — they are anagrams, so we skip "bbaa".
4. Compare "cd" with "bbaa" — they are not anagrams, so we add "cd" to `result`.
5. Compare "cd" with "cd" — they are anagrams, so we skip the second "cd".

Final output: `["abba", "cd"]`.

#### Example 2:
```cpp
Input: words = ["a", "b", "c", "d", "e"]
```

- Since none of the words are anagrams of each other, no words are removed.

Final output: `["a", "b", "c", "d", "e"]`.

### Time Complexity:
- **isAnagram Function:** For each pair of words, we process each character, so it takes \(O(m)\), where \(m\) is the length of the word. In the worst case, we compare each word with its previous word.
- **Overall Time Complexity:** \(O(n \times m)\), where \(n\) is the number of words and \(m\) is the maximum length of a word.

### Space Complexity:
- We use an extra `count[26]` array to store the frequency of characters for each word, which requires \(O(1)\) space because the array always has 26 elements.
- The space complexity for storing the result is \(O(n)\), where \(n\) is the number of words in the input list.

This approach efficiently solves the problem within the given constraints.