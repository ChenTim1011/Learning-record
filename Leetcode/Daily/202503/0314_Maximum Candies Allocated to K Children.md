[Maximum Candies Allocated to K Children](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/description/?envType=daily-question&envId=2025-03-14)

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int maximumCandies(vector<int>& candies, long long k) {
        long long left = 1, right = *max_element(candies.begin(), candies.end());
        int ans = 0;

        while (left <= right) {
            long long mid = left + (right - left) / 2;
            
            if (canDistribute(candies, k, mid)) {
                ans = mid;  // Store the maximum valid answer
                left = mid + 1;  // Try for a bigger candy count
            } else {
                right = mid - 1;  // Reduce the search space
            }
        }
        
        return ans;
    }

private:
    bool canDistribute(vector<int>& candies, long long k, long long mid) {
        long long count = 0;
        for (int c : candies) {
            count += c / mid;  // Count how many groups of `mid` can be formed
            if (count >= k) return true;  // If we can serve all children, return true
        }
        return false;
    }
};
