[House Robber IV](https://leetcode.com/problems/house-robber-iv/description/?envType=daily-question&envId=2025-03-15)

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minCapability(vector<int>& nums, int k) {
        int left = *min_element(nums.begin(), nums.end());
        int right = *max_element(nums.begin(), nums.end());
        int ans = right;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (canRob(nums, k, mid)) {
                ans = mid;  // Try for a smaller capability
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }

private:
    bool canRob(vector<int>& nums, int k, int maxCap) {
        int count = 0;
        int n = nums.size();

        for (int i = 0; i < n; i++) {
            if (nums[i] <= maxCap) {
                count++;
                i++;  // Skip next house (since adjacent houses cannot be robbed)
            }
        }
        return count >= k;
    }
};
