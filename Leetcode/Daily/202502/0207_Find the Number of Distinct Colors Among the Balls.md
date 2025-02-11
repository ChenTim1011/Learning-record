[Find the Number of Distinct Colors Among the Balls](https://leetcode.com/problems/find-the-number-of-distinct-colors-among-the-balls/description/)


```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    /**
     * This function processes a series of queries that paint balls with colors.
     * After each query, it returns the number of distinct colors present.
     *
     * @param limit   The highest index of a ball (balls are labeled from 0 to limit).
     * @param queries A 2D vector where each inner vector represents a query in the form [ball, color].
     * @return        A vector containing the number of distinct colors after each query.
     */
    vector<int> queryResults(int limit, vector<vector<int>>& queries) {
        // 'ball' map: stores the current color of each ball.
        // Key: ball index, Value: current color assigned to that ball.
        unordered_map<int, int> ball;
        
        // 'color' map: stores the frequency (number of balls) for each color.
        // Key: color, Value: count of balls that currently have that color.
        unordered_map<int, int> color;
        
        // 'ans' vector will hold the result after each query.
        vector<int> ans;
        ans.reserve(queries.size());  // Reserve space for efficiency.
        
        // 'distinct' variable keeps track of the number of distinct colors currently used.
        int distinct = 0;
        
        // Process each query in the 'queries' vector.
        for (auto &q : queries) {
            // Extract the ball position (label) and the new color from the current query.
            int pos = q[0];  // Ball index.
            int c = q[1];    // New color to assign to the ball.
            
            // Check if this ball has been colored before.
            if (ball.count(pos)) {
                // If the ball already has a color, decrement the frequency of its old color.
                if (--color[ball[pos]] == 0) {
                    // If the frequency becomes zero, no ball is using that color anymore.
                    // Remove the color from the 'color' map.
                    color.erase(ball[pos]);
                    // Decrement the count of distinct colors.
                    distinct--;
                }
            }
            
            // Paint the ball with the new color by updating the 'ball' map.
            ball[pos] = c;
            
            // Increment the frequency for the new color.
            // If the new color did not exist before (i.e., frequency becomes 1), it is a new distinct color.
            if (++color[c] == 1)
                distinct++;
            
            // Append the current count of distinct colors to the answer vector.
            ans.push_back(distinct);
        }
        
        // Return the results after processing all queries.
        return ans;
    }
};

int main() {
    // Create an instance of the solution class.
    Solution solution;
    
    // Example input:
    // 'limit' represents the highest ball label.
    // 'queries' is a list of operations where each operation is [ball, color].
    int limit = 4;
    vector<vector<int>> queries = {
        {1, 4}, // Paint ball 1 with color 4.
        {2, 5}, // Paint ball 2 with color 5.
        {1, 3}, // Repaint ball 1 from color 4 to color 3.
        {3, 4}  // Paint ball 3 with color 4.
    };
    
    // Get the result vector after processing all queries.
    vector<int> result = solution.queryResults(limit, queries);
    
    // Output the result:
    // Each number represents the number of distinct colors after each respective query.
    cout << "Number of distinct colors after each query: ";
    for (int count : result) {
        cout << count << " ";
    }
    cout << endl;
    
    return 0;
}
```

---

### Detailed Explanation of the Code

1. **Include Libraries:**
   - `<iostream>` is used for input and output.
   - `<vector>` is used to handle dynamic arrays.
   - `<unordered_map>` is used for the hash maps that store ball colors and color frequencies.

2. **Solution Class and `queryResults` Method:**
   - **Ball Map (`ball`):**  
     Keeps track of the current color for each ball.
   - **Color Map (`color`):**  
     Maintains a frequency count of each color currently used.
   - **Answer Vector (`ans`):**  
     Stores the number of distinct colors after each query.
   - **Processing Queries:**  
     For each query:
     - Check if the ball already has a color. If yes, decrement the count for that color. If the count reaches zero, remove that color and decrease the distinct count.
     - Update the ball's color with the new color from the query.
     - Increase the frequency count for the new color. If it is the first time the color appears, increase the distinct count.
     - Append the current number of distinct colors to `ans`.

3. **Main Function:**
   - An instance of the `Solution` class is created.
   - Sample input is defined.
   - The `queryResults` method is called to process the queries.
   - The result is printed out, showing the number of distinct colors after each query.

