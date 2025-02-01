[Minimum Length of String After Operations](https://leetcode.com/problems/minimum-length-of-string-after-operations/description/?envType=daily-question&envId=2025-01-13)



```c++
class Solution {
public:
    int minimumLength(string s) {
        unordered_map<char, int> count;
        for (char c : s) count[c]++;
        int minus = 0;
        for (auto& entry : count) {
            while (entry.second >= 3) {
                minus += 2;
                entry.second -= 2;
            }
        }
        return s.length() - minus;
    }
};
```


```c++
class Solution {
public:
    int minimumLength(string s) {
        vector<int> charFrequency(26, 0);
        int totalLength = 0;
        for (char currentChar : s) {
            charFrequency[currentChar - 'a']++;
        }
        for (int frequency : charFrequency) {
            if (frequency == 0) continue;
            if (frequency % 2 == 0) {
                totalLength += 2;
            } else {
                totalLength += 1;
            }
        }
        return totalLength;
    }
};
```

In summary, because the operation in the problem only requires that a character has at least one identical neighbor on each side (a condition that is automatically satisfied if the frequency is high enough), we can determine the minimum final length of the string simply by counting the frequency of each character, without considering the specific positions.