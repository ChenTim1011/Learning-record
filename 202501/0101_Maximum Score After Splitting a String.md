[Maximum Score After Splitting a String](https://leetcode.com/problems/maximum-score-after-splitting-a-string/description/?envType=daily-question&envId=2025-01-01)

## Brute force  with substr
```c++
class Solution {
public:
    int maxScore(string s) {
        int result = INT_MIN; // Initialize result to the smallest possible integer
     
        // Iterate over each possible split point
        for(int i = 0; i < s.size() - 1; i++) {
            int count1 = 0; // Count of '0's in the left substring
            int count2 = 0; // Count of '1's in the right substring
            string str1 = s.substr(0, i + 1); // Left substring
            string str2 = s.substr(i + 1, s.size() - (i + 1)); // Right substring
            
            // Count '0's in the left substring
            for(int j = 0; j < str1.size(); j++) {
                if(str1[j] == '0') {
                    count1++;
                }
            }
            
            // Count '1's in the right substring
            for(int j = 0; j < str2.size(); j++) {
                if(str2[j] == '1') {
                    count2++;
                }
            }
            
            // Update the result with the maximum score
            result = max(result, count1 + count2);
        }
        return result; // Return the maximum score
    }
};
```

## Brute force without substr

```c++
class Solution {
public:
    int maxScore(string s) {
        int result = INT_MIN; // Initialize result to the smallest possible integer
     
        // Iterate over each possible split point
        for(int i = 0; i < s.size() - 1; i++) {
            int count1 = 0; // Count of '0's in the left substring
            int count2 = 0; // Count of '1's in the right substring
            
            // Count '0's in the left substring
            for(int j = 0; j <= i; j++) {
                if(s[j] == '0') {
                    count1++;
                }
            }
            
            // Count '1's in the right substring
            for(int j = i + 1; j < s.size(); j++) {
                if(s[j] == '1') {
                    count2++;
                }
            }
            
            // Update the result with the maximum score
            result = max(result, count1 + count2);
        }
        return result; // Return the maximum score
    }
};
```

## O(n) solution
```c++
class Solution {
public:
    int maxScore(string s) {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);

        int n = s.size(); // Get the size of the string
        int maxs = 0; // Initialize the maximum score to 0
        
        // Precompute the number of '0's to the left of each position
        vector<int> leftZero(n, 0), rightOne(n, 0);

        // Calculate the cumulative '0's on the left side
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                leftZero[i] = leftZero[i - 1];
            }
            if (s[i] == '0') {
                leftZero[i]++;
            }
        }

        // Calculate the cumulative '1's on the right side
        for (int i = n - 1; i >= 0; i--) {
            if (i < n - 1) {
                rightOne[i] = rightOne[i + 1];
            }
            if (s[i] == '1') {
                rightOne[i]++;
            }
        }

        // Iterate over all possible split points and calculate the maximum score
        for (int i = 1; i < n; i++) { // Split point must divide the string into two non-empty parts
            int sum = leftZero[i - 1] + rightOne[i];
            maxs = max(maxs, sum);
        }

        return maxs; // Return the maximum score
    }
};
```

Explanation:
- **Brute force**: This approach iterates over each possible split point, creates substrings, and counts '0's in the left substring and '1's in the right substring. It then updates the result with the maximum score found.
- **Brute force without substr**: This approach is similar to the brute force method but avoids creating substrings by directly counting '0's and '1's in the respective parts of the string.
- **O(n) solution**: This optimized approach precomputes the number of '0's to the left of each position and the number of '1's to the right of each position. It then iterates over all possible split points to calculate the maximum score efficiently.

The `O(n)` solution is more efficient because it avoids redundant calculations by using precomputed arrays for cumulative counts of '0's and '1's. This reduces the time complexity from `O(n^2)` to `O(n)`.