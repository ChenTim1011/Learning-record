[ Bitwise XOR of All Pairings](https://leetcode.com/problems/bitwise-xor-of-all-pairings/description/)

# Wrong solution (TLE)
```c++
class Solution {
public:
    int xorAllNums(vector<int>& nums1, vector<int>& nums2) {
        int result = 0;
        for(int i=0;i<nums1.size();i++){
            for(int j=0;j<nums2.size();j++){
                result = result ^ (nums1[i]^nums2[j]);
            }
        }
        return result;
    }
};
```

```c++
class Solution {
public:
    int xorAllNums(vector<int>& nums1, vector<int>& nums2) {
        int result = 0;

        // If the size of nums1 is odd, each element in nums2 will be XORed an odd number of times
        if (nums1.size() % 2 != 0) {
            for (int num : nums2) {
                result ^= num; // XOR each element of nums2 to the result
            }
        }

        // If the size of nums2 is odd, each element in nums1 will be XORed an odd number of times
        if (nums2.size() % 2 != 0) {
            for (int num : nums1) {
                result ^= num; // XOR each element of nums1 to the result
            }
        }

        return result;
    }
};

Detailed Explanation:
Initialization:

result is initialized to 0. This variable will store the cumulative XOR of all pairings.
Odd Size of nums1:

If the size of nums1 is odd (nums1.size() % 2 != 0), each element in nums2 will be XORed an odd number of times in the nested loop of the original solution.
XORing a number an odd number of times results in the number itself (e.g., x ^ x ^ x = x).
Therefore, we XOR each element of nums2 to result.
Odd Size of nums2:

Similarly, if the size of nums2 is odd (nums2.size() % 2 != 0), each element in nums1 will be XORed an odd number of times in the nested loop of the original solution.
We XOR each element of nums1 to result.
Return Result:

The final value of result is returned, which is the XOR of all pairings.
Time Complexity:
The time complexity of this solution is O(n + m), where n is the size of nums1 and m is the size of nums2. This is because we iterate through each array at most once.
Space Complexity:
The space complexity is O(1) as we are using only a constant amount of extra space.