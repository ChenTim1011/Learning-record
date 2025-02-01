[Make Lexicographically Smallest Array by Swapping Elements](https://leetcode.com/problems/make-lexicographically-smallest-array-by-swapping-elements/description/)

```c++
#include <vector>
#include <algorithm>
#include <utility>

using namespace std;

class Solution {
public:
    vector<int> lexicographicallySmallestArray(vector<int>& arr, int threshold) {
        vector<pair<int, int>> valueIndexPairs; // To store each element and its original index
        int size = arr.size();

        // Step 1: Pair each element with its index and store it in valueIndexPairs
        for (int i = 0; i < size; ++i) {
            valueIndexPairs.push_back({arr[i], i});
        }

        // Step 2: Sort valueIndexPairs by value to consider elements in ascending order
        sort(valueIndexPairs.begin(), valueIndexPairs.end());

        vector<vector<pair<int, int>>> groupedPairs; // To store groups of elements with differences <= threshold
        groupedPairs.push_back({valueIndexPairs[0]}); // Start the first group with the first element

        // Step 3: Group elements that can be swapped based on the threshold
        for (int i = 1; i < size; ++i) {
            if (valueIndexPairs[i].first - valueIndexPairs[i - 1].first <= threshold) {
                // If the difference between consecutive elements is within the threshold, add to the current group
                groupedPairs.back().push_back(valueIndexPairs[i]);
            } else {
                // Otherwise, start a new group
                groupedPairs.push_back({valueIndexPairs[i]});
            }
        }

        // Step 4: Sort the indices within each group and update the original array
        for (const auto& group : groupedPairs) {
            vector<int> indices; // Store the indices of the current group
            for (const auto& [value, index] : group) {
                indices.push_back(index);
            }

            // Sort indices to maintain lexicographical order in the original array
            sort(indices.begin(), indices.end());

            // Update the array with the sorted values in their respective indices
            for (size_t i = 0; i < indices.size(); ++i) {
                arr[indices[i]] = group[i].first;
            }
        }

        return arr; // Return the modified array
    }
};
```

/*
### Explanation of the Steps:

1. **Pair Elements with Indices:**
   We need to keep track of the original positions of elements in the array so we can reassign values to the correct indices after sorting and grouping.

2. **Sort Elements:**
   Sorting the value-index pairs ensures that we process elements in ascending order. This helps us group elements based on their value differences.

3. **Group Elements:**
   Using the threshold, we group consecutive elements that can be swapped (i.e., their difference is <= threshold).

4. **Sort Within Groups:**
   Within each group, we sort the indices to ensure the elements are placed in the smallest lexicographical order possible.

5. **Update the Original Array:**
   After sorting within each group, we update the array with the sorted values, maintaining the original indices.

### Complexity Analysis:

- **Time Complexity:**
  - Sorting the `valueIndexPairs` takes O(n log n).
  - Grouping and sorting indices within each group takes O(n log n) in total.
  - Overall complexity: O(n log n).

- **Space Complexity:**
  - Storing the `valueIndexPairs` and `groupedPairs` requires O(n) additional space.
*/