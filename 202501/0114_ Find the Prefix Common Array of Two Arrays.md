[Find the Prefix Common Array of Two Arrays](https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/description/)

## Brute solution

```c++
class Solution {
public:
    vector<int> findThePrefixCommonArray(vector<int>& A, vector<int>& B) {
        // form back to front  the last one will be the length of A or B
        // A = [2,3,1], B = [3,1,2]
        int n = A.size();
        int len = n;
        vector<int> result(n,0);
        result[n-1]=len;
        for(int i=n-2;i>=0;i--){
            int count = 0 ;
            for(int j=0;j<=i;j++){
                for(int k=0;k<=i;k++){
                    if(A[j]==B[k]){
                        count++;
                    }
                }
            }
            result[i]=count;
        }
        return result;
    }
};
```
I'll help you create a more efficient solution to this problem. The current solution has O(n³) time complexity due to the nested loops, which isn't optimal. Let's create a better solution with O(n) time complexity.



```cpp
class Solution {
public:
    vector<int> findThePrefixCommonArray(vector<int>& A, vector<int>& B) {
        int n = A.size();
        vector<int> result(n);
        vector<bool> seenA(n + 1, false);  // Track numbers seen in A
        vector<bool> seenB(n + 1, false);  // Track numbers seen in B
        int common = 0;
        
        // Process both arrays simultaneously
        for(int i = 0; i < n; i++) {
            // Process current number from A
            seenA[A[i]] = true;
            if(seenB[A[i]]) {
                common++;
            }
            
            // Process current number from B
            seenB[B[i]] = true;
            if(seenA[B[i]] && A[i] != B[i]) {  // Don't count same number twice
                common++;
            }
            
            result[i] = common;
        }
        
        return result;
    }
};

## Better solution

```c++
class Solution {
public:
    vector<int> findThePrefixCommonArray(vector<int>& A, vector<int>& B) {
        int n = A.size();
        vector<int> result(n);
        vector<int> freq(n + 1, 0);  // frequency array to track numbers
        int commonCount = 0;
        
        // Process both arrays simultaneously
        for(int i = 0; i < n; i++) {
            // Process number from A
            freq[A[i]]++;
            if(freq[A[i]] == 2) commonCount++;
            
            // Process number from B
            freq[B[i]]++;
            if(freq[B[i]] == 2) commonCount++;
            
            result[i] = commonCount;
        }
        
        return result;
    }
};
```

Let's break down how this works:

We create a frequency array freq to keep track of how many times we've seen each number.
We process both arrays A and B simultaneously index by index.
For each index i:

When we see a number in A, we increment its frequency
If its frequency becomes 2, we found a common number
We do the same for the number in B
The result at index i is our running count of common numbers



Time Complexity: O(n) - we only need one pass through both arrays
Space Complexity: O(n) - we use additional space for the frequency array
Let's trace Example 1: A = [1,3,2,4], B = [3,1,2,4]
Copyi = 0:
- Process A[0]=1: freq[1]=1
- Process B[0]=3: freq[3]=1
- result[0]=0

i = 1:
- Process A[1]=3: freq[3]=2 (commonCount++)
- Process B[1]=1: freq[1]=2 (commonCount++)
- result[1]=2

i = 2:
- Process A[2]=2: freq[2]=1
- Process B[2]=2: freq[2]=2 (commonCount++)
- result[2]=3

i = 3:
- Process A[3]=4: freq[4]=1
- Process B[3]=4: freq[4]=2 (commonCount++)
- result[3]=4
This solution is more efficient than the original O(n³) solution and easier to understand. It processes each element exactly once and uses a frequency array to track common numbers, making it both time and space efficient.