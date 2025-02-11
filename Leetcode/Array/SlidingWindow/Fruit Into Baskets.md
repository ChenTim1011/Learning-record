[ Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/)

# Sliding windows

```c++
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        int n = fruits.size(); // Get the size of the fruits array
        map<int, int> mpp; // To store the count of each fruit type in the window
        int l = 0, r = 0;  // Sliding window pointers
        int maxi = 0;      // Maximum size of the window

        // Iterate over the array with the right pointer
        while (r < n) {
            mpp[fruits[r]]++; // Add the current fruit to the map and increment its count

            // If more than two types of fruits are in the window
            while (mpp.size() > 2) {
                mpp[fruits[l]]--; // Decrement the count of the leftmost fruit
                if (mpp[fruits[l]] == 0) mpp.erase(fruits[l]); // Remove the fruit type from the map if its count is zero
                l++; // Shrink the window from the left
            }

            // Update the maximum size of the window
            maxi = max(maxi, r - l + 1);
            r++; // Expand the window to the right
        }
        return maxi; // Return the maximum size of the window
    }
};
```

Explanation of `map` usage:
- `map<int, int> mpp;` initializes a map to store the count of each fruit type in the current window.
- `mpp[fruits[r]]++;` increments the count of the current fruit type.
- `while (mpp.size() > 2)` checks if there are more than two types of fruits in the window.
- `mpp[fruits[l]]--;` decrements the count of the leftmost fruit type.
- `if (mpp[fruits[l]] == 0) mpp.erase(fruits[l]);` removes the fruit type from the map if its count is zero.