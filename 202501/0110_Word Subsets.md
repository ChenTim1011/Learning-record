[Word Subsets](https://leetcode.com/problems/word-subsets/description/)

## My solution: Brute force method
```c++
class Solution {
public:
    vector<string> wordSubsets(vector<string>& words1, vector<string>& words2) {
   
        int maxrequirewords2[26]={0};
        bool check = true;
        for(int i=0;i<words2.size();i++){
            int countwords2[26]={0};
            for(int j=0;j<words2[i].size();j++){
                countwords2[words2[i][j]-'a']++;
            }     
            for(int k=0;k<26;k++){
                maxrequirewords2[k]=max(maxrequirewords2[k],countwords2[k]);
            }
        }
        vector<string> result;
        for(int i=0;i<words1.size();i++){
            int countwords1[26]={0};
            for(int j=0;j<words1[i].size();j++){
                countwords1[words1[i][j]-'a']++;
            }
        
            check = true;
            for(int k=0;k<26;k++){
                if(countwords1[k]<maxrequirewords2[k]){
                    check=false;
                    break;
                }
            }
            if(check==true){
                result.emplace_back(words1[i]);
            }
        }
        return result;
    }
};
```

Below is the code with comments and an explanation of how this method works.

### Code with Comments

```cpp
class Solution {
public:
    vector<string> wordSubsets(vector<string>& mainWords, vector<string>& requiredWords) {
        int maxCharFreq[26] = {0}; // To store the maximum frequency of each character across all words in requiredWords.
        int tempCharFreq[26];      // Temporary array to store the character frequency for each word.

        // Step 1: Calculate the maximum frequency for each character in requiredWords.
        for (const auto& word : requiredWords) {
            memset(tempCharFreq, 0, sizeof tempCharFreq); // Reset tempCharFreq to 0 for each word.
            for (char ch : word) {
                tempCharFreq[ch - 'a']++; // Count frequency of each character in the current word.
            }
            for (int i = 0; i < 26; ++i) {
                // Update maxCharFreq to store the maximum frequency for each character across all requiredWords.
                maxCharFreq[i] = max(maxCharFreq[i], tempCharFreq[i]);
            }
        }

        vector<string> universalWords; // To store the result, i.e., all universal words.

        // Step 2: Check each word in mainWords to see if it satisfies the "universal" condition.
        for (const auto& word : mainWords) {
            memset(tempCharFreq, 0, sizeof tempCharFreq); // Reset tempCharFreq to 0 for each word in mainWords.
            for (char ch : word) {
                tempCharFreq[ch - 'a']++; // Count frequency of each character in the current word.
            }
            bool isUniversal = true; // Assume the word is universal initially.
            for (int i = 0; i < 26; ++i) {
                // Check if the current word has at least the required frequency for each character.
                if (maxCharFreq[i] > tempCharFreq[i]) {
                    isUniversal = false; // If not, mark it as not universal and break.
                    break;
                }
            }
            if (isUniversal) {
                universalWords.emplace_back(word); // If the word is universal, add it to the result.
            }
        }

        return universalWords; // Return all universal words.
    }
};
```

---

### Explanation

This method solves the problem of finding **universal words** in `mainWords` relative to `requiredWords`. A word in `mainWords` is considered universal if it contains all the characters in `requiredWords` with the required frequency. 

#### Steps Breakdown

1. **Preprocessing `requiredWords`:**
   - For each word in `requiredWords`, calculate the frequency of each character using `tempCharFreq`.
   - Maintain a global frequency `maxCharFreq` to store the maximum frequency of each character across all `requiredWords`. This ensures that all the constraints from `requiredWords` are consolidated into a single frequency array.

2. **Processing `mainWords`:**
   - For each word in `mainWords`, calculate its character frequencies using `tempCharFreq`.
   - Compare these frequencies against `maxCharFreq`. If the word satisfies all the character frequency constraints from `maxCharFreq`, it is marked as universal and added to the result.

3. **Return Result:**
   - Return all the universal words stored in the `universalWords` vector.

#### Key Concepts Used

- **Character Frequency Array:** The `maxCharFreq` array represents the frequency requirements for each character ('a' to 'z'). Using a fixed-size array of 26 ensures efficient and fast access for character counting.
- **Resetting Frequencies:** `memset` is used to reset the temporary frequency array (`tempCharFreq`) to 0 before processing a new word. This is faster than creating a new array each time.
- **Efficiency:**
  - The solution processes each word in `mainWords` and `requiredWords` linearly with respect to their lengths.
  - The comparison of character frequencies (26 iterations for each word) is constant time.

---

### Time Complexity

- **Processing `requiredWords`:** \(O(L_r)\), where \(L_r\) is the total number of characters in all `requiredWords`.
- **Processing `mainWords`:** \(O(L_m)\), where \(L_m\) is the total number of characters in all `mainWords`.
- **Comparison of Frequencies:** \(O(1)\) for each word due to the fixed size of the frequency array.

**Overall Complexity:** \(O(L_r + L_m)\), which is very efficient.

---

### Space Complexity

- The solution uses:
  - Two fixed-size arrays of size 26: `maxCharFreq` and `tempCharFreq`.
  - A result vector to store the universal words.
  
**Overall Space Complexity:** \(O(U + 26)\), where \(U\) is the number of universal words in `mainWords`.