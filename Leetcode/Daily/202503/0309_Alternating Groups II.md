[Alternating Groups II](https://leetcode.com/problems/alternating-groups-ii/description/?envType=daily-question&envId=2025-03-09)

## **📌 Problem Statement**
We are given:
1. **A string `word`** (consisting of lowercase English letters).
2. **An integer `k`**, representing the exact number of consonants in a valid substring.

We need to **count substrings** that:
1. Contain **all vowels (`a, e, i, o, u`) at least once**.
2. Have **exactly `k` consonants**.

---

## **🔹 Key Observations**
1. **Sliding Window Approach**:
   - Since we are looking for contiguous substrings, a **two-pointer sliding window** is ideal.
   - Expand the window by moving `right`.
   - Shrink the window by moving `left` to maintain `k` consonants.

2. **Tracking Vowels and Consonants Efficiently**:
   - Maintain a **frequency table** for vowels.
   - Use `currentK` to track **the number of consonants** in the window.
   - Use `vowels` to track **how many distinct vowels are in the window**.

3. **Edge Cases**:
   - If `word` is too short (`word.length < 5`), it's **impossible** to contain all vowels.
   - If `k = 0`, we only count substrings containing only vowels.

---

## **🔹 Approach**
1. **Initialize a frequency table** to track vowels and consonants.
2. **Use a sliding window (`left`, `right`)**:
   - Expand `right`, updating vowel/consonant count.
   - If `currentK > k`, shrink `left` to restore validity.
   - Count valid substrings when **all vowels exist and consonants count equals `k`**.
3. **Handle Overlapping Cases**:
   - If multiple substrings are valid, count them efficiently.

---

## **💻 Code Implementation**
```cpp
class Solution {
public:
    long long countOfSubstrings(string word, int k) {
        int frequencies[2][128] = {};  
        frequencies[0]['a'] = frequencies[0]['e'] = frequencies[0]['i'] = frequencies[0]['o'] = frequencies[0]['u'] = 1;

        long long response = 0;
        int currentK = 0, vowels = 0, extraLeft = 0, left = 0;

        for (int right = 0; right < word.length(); right++) {
            char rightChar = word[right];

            // Check if rightChar is a vowel
            if (frequencies[0][rightChar]) {  
                if (++frequencies[1][rightChar] == 1) vowels++;  
            } else { 
                currentK++; // It's a consonant
            }

            // Shrink window when consonant count exceeds k
            while (currentK > k) {
                char leftChar = word[left++];
                if (frequencies[0][leftChar]) {  
                    if (--frequencies[1][leftChar] == 0) vowels--;  
                } else {
                    currentK--;
                }
                extraLeft = 0;
            }

            // Minimize left to remove unnecessary leading vowels
            while (vowels == 5 && currentK == k && left < right && frequencies[0][word[left]] && frequencies[1][word[left]] > 1) {
                extraLeft++;
                frequencies[1][word[left++]]--;
            }

            // Count valid substrings
            if (currentK == k && vowels == 5) {
                response += (1 + extraLeft);
            }
        }

        return response;
    }
};
```

---

## **⏳ Complexity Analysis**
| **Operation**   | **Time Complexity** | **Space Complexity** |
|----------------|--------------------|--------------------|
| **Sliding Window Traversal** | **O(n)** | **O(1)** |

- **Time Complexity: O(n)**  
  - Each character is **processed at most twice** (once when expanding `right`, once when shrinking `left`).
- **Space Complexity: O(1)**  
  - We use **fixed-sized arrays** (`128` size) to track frequencies.

---

## **✅ Summary**
| Approach | Time Complexity | Space Complexity | Notes |
|----------|---------------|----------------|----------------|
| **Sliding Window** | **O(n)** | **O(1)** | Efficient, avoids nested loops. |

