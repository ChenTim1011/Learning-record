[Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/description/)

### Problem Explanation

The task is to determine whether a course is a prerequisite for another course given:
- **`numCourses`**: The total number of courses (numbered from `0` to `numCourses - 1`).
- **`prerequisites`**: A list of prerequisite pairs, where each pair `[a, b]` indicates that course `a` must be taken before course `b`.
- **`queries`**: A list of pairs `[u, v]`, asking whether course `u` is a prerequisite for course `v`.

---

### Intuition

Think of this as constructing a "family tree" of courses:
1. Direct prerequisites are like parents.
2. Indirect prerequisites are like grandparents or ancestors.

We need to check for every query whether a course `u` is an ancestor of course `v` in this dependency graph.

---

### Approach

To solve this efficiently, we can:
1. Build a **graph** to represent the dependencies between courses.
2. Use **topological sorting** (via BFS) to determine all the prerequisites (direct and indirect) for every course.
3. Use a **map** to store the prerequisites for each course.
4. Answer the queries in constant time by checking if a course exists in the prerequisite set of another course.

---

### Steps to Solve

1. **Build the graph**:
   - Use an adjacency list to represent dependencies between courses.
   - Track the number of prerequisites for each course using an indegree array.

2. **Topological sorting using BFS**:
   - Start with courses that have no prerequisites (indegree = 0).
   - For each course, process its neighbors and update their prerequisites using the map.

3. **Answer queries**:
   - For each query `[u, v]`, check if course `u` is in the prerequisite set of course `v` (stored in the map).

---

### Code with Explanation

```cpp
class Solution {
public:
    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        // Step 1: Build graph and indegree array
        vector<vector<int>> adj(numCourses); // adjacency list
        vector<int> indegree(numCourses, 0); // indegree array to track prerequisites count
        
        for (auto p : prerequisites) {
            adj[p[0]].push_back(p[1]); // p[0] -> p[1]
            indegree[p[1]]++;          // Increment indegree of p[1]
        }
        
        // Step 2: Perform topological sort (BFS) and track prerequisites
        queue<int> q; // queue to process nodes with indegree 0
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                q.push(i); // Add courses with no prerequisites
            }
        }
        
        unordered_map<int, unordered_set<int>> mp; // Map to store all prerequisites for each course
        
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            
            // Process all neighbors of the current course
            for (auto next : adj[curr]) {
                mp[next].insert(curr); // Add the current course as a direct prerequisite
                
                // Add all indirect prerequisites from `curr` to `next`
                for (auto pre : mp[curr]) {
                    mp[next].insert(pre);
                }
                
                // Decrease the indegree of the neighbor
                indegree[next]--;
                // If the neighbor has no more prerequisites, add it to the queue
                if (indegree[next] == 0) {
                    q.push(next);
                }
            }
        }
        
        // Step 3: Answer the queries
        vector<bool> res;
        for (auto q : queries) {
            res.push_back(mp[q[1]].count(q[0])); // Check if q[0] is a prerequisite of q[1]
        }
        return res;
    }
};
```

---

### Explanation of Code with Comments

1. **Graph Construction**:
   ```cpp
   vector<vector<int>> adj(numCourses); 
   vector<int> indegree(numCourses, 0); 
   
   for (auto p : prerequisites) {
       adj[p[0]].push_back(p[1]); // Create edge p[0] -> p[1]
       indegree[p[1]]++;          // Increment indegree of p[1]
   }
   ```
   - `adj` stores the directed graph, where each course points to the courses that depend on it.
   - `indegree` keeps track of how many prerequisites each course has.

2. **Topological Sorting**:
   ```cpp
   queue<int> q;
   for (int i = 0; i < numCourses; i++) {
       if (indegree[i] == 0) {
           q.push(i); // Courses with no prerequisites
       }
   }
   ```
   - Initialize the queue with all courses that have no prerequisites (`indegree = 0`).

3. **Tracking Prerequisites**:
   ```cpp
   while (!q.empty()) {
       int curr = q.front();
       q.pop();
       
       for (auto next : adj[curr]) {
           mp[next].insert(curr); // Add curr as a prerequisite for next
           for (auto pre : mp[curr]) {
               mp[next].insert(pre); // Add indirect prerequisites
           }
           indegree[next]--;
           if (indegree[next] == 0) {
               q.push(next); // Add course with no more prerequisites
           }
       }
   }
   ```
   - For each course, add it as a prerequisite for its dependent courses (`next`).
   - Inherit all prerequisites from the current course.

4. **Answering Queries**:
   ```cpp
   vector<bool> res;
   for (auto q : queries) {
       res.push_back(mp[q[1]].count(q[0]));
   }
   ```
   - For each query, check if `q[0]` exists in the prerequisite set of `q[1]`.

---

### Complexity Analysis

1. **Time Complexity**:
   - **Building the graph**: \( O(P) \), where \( P \) is the number of prerequisites.
   - **Topological sort**: \( O(V + E) \), where \( V \) is the number of courses and \( E \) is the number of edges.
   - **Answering queries**: \( O(Q) \), where \( Q \) is the number of queries.
   - **Total**: \( O(V + E + Q) \).

2. **Space Complexity**:
   - **Adjacency list**: \( O(E) \).
   - **Queue**: \( O(V) \).
   - **Map**: \( O(V \times E) \), in the worst case where every course depends on every other course.
   - **Total**: \( O(V \times E) \).

---

### Example Walkthrough

#### Input:
- `numCourses = 4`
- `prerequisites = [[0, 1], [1, 2], [2, 3]]`
- `queries = [[0, 1], [1, 3], [3, 0], [2, 3]]`

#### Process:
1. **Graph**:
   ```
   0 -> 1 -> 2 -> 3
   ```

2. **Topological Sort**:
   - `mp[1] = {0}`
   - `mp[2] = {0, 1}`
   - `mp[3] = {0, 1, 2}`

3. **Answer Queries**:
   - `[0, 1]`: `true`
   - `[1, 3]`: `true`
   - `[3, 0]`: `false`
   - `[2, 3]`: `true`

#### Output:
```
[true, true, false, true]
```

---
