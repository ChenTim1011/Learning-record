[Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)

## Map method without using priority_queue 

```c++
class Solution {

public:
    // Custom comparator to sort pairs by their second value (frequency) in descending order
    static bool comp(pair<int, int>& a, pair<int, int>& b) {
        return a.second > b.second;
    }

    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> map; // Map to store the frequency of each element
        for(int i = 0; i < nums.size(); i++) {
            map[nums[i]]++; // Increment the frequency of the current element
        }
        
        vector<pair<int, int>> vec(map.begin(), map.end()); // Convert map to a vector of pairs

        sort(vec.begin(), vec.end(), comp); // Sort the vector by frequency in descending order
        vector<int> result;
        for(int i = 0; i < k; i++) {
            result.push_back(vec[i].first); // Add the top k frequent elements to the result
        }
        
        return result; // Return the result vector
    }
};
```

## priority_queue method

```c++
class Solution {
public:
    // Custom comparator for the priority queue to implement a min-heap
    class mycmp {
    public:
        bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
            return lhs.second > rhs.second; // Compare by frequency in ascending order
        }
    };

    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> maps; // Map to store the frequency of each element
        // nums = [1,1,1,2,2,3], k = 2
        for(int i = 0; i < nums.size(); i++) {
            maps[nums[i]]++; // Increment the frequency of the current element
        }      
        // map 1:3 2:2 3:1
        priority_queue<pair<int, int>, vector<pair<int, int>>, mycmp> pe; // Min-heap to store the top k elements
        for(auto it = maps.begin(); it != maps.end(); it++) {  
            pe.push(*it); // Push the current element and its frequency into the heap
            if(pe.size() > k) {
                pe.pop(); // If the heap size exceeds k, remove the element with the smallest frequency
            }
        }
        vector<int> result(k, 0);
        for(int i = k - 1; i >= 0; i--) {
            result[i] = pe.top().first; // Extract the top k elements from the heap
            pe.pop();
        }
        return result; // Return the result vector
    }
};
```

---

### **Code Analysis and Explanation**

#### **1. Outer Class Definition**
```cpp
class Solution {
public:
```
- `Solution` is a public class that allows external access to its member functions.

---

#### **2. Define Inner Class `mycmp`**
```cpp
class mycmp {
public:
    bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
        return lhs.second > rhs.second;
    }
};
```
- **Purpose**: Defines a custom comparator for the priority queue to implement a min-heap.
- **Key Details**:
  - `operator()` overloads the comparison operator.
  - Parameters: `lhs` and `rhs` are two `pair<int, int>` elements, representing a number and its frequency.
  - Logic: 
    - Returns `true` if `lhs.second > rhs.second`, meaning `rhs` has higher priority (lower frequency is prioritized).
    - Effect: The priority queue becomes a **min-heap** where elements are ordered by frequency in ascending order.

---

#### **3. Main Function `topKFrequent`**
```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
```
- **Purpose**: Returns the top `k` most frequent elements in the array.
- **Parameters**:
  - `nums`: The input array of integers.
  - `k`: The number of most frequent elements to return.

---

#### **4. Count Frequency Using `unordered_map`**
```cpp
unordered_map<int, int> map; // map<nums[i], frequency>
for (int i = 0; i < nums.size(); i++) {
    map[nums[i]]++;
}
```
- Uses an `unordered_map` to store the frequency of each number in `nums`.
- Example:
  - Input: `nums = [1, 1, 1, 2, 2, 3]`.
  - Resulting `map`: `{1: 3, 2: 2, 3: 1}`.

---

#### **5. Define a Min-Heap**
```cpp
priority_queue<pair<int, int>, vector<pair<int, int>>, mycmp> pri_que;
```
- **Purpose**: Implements a min-heap to store the top `k` elements by frequency.
- **Key Components**:
  - `pair<int, int>`: Stores the number (`first`) and its frequency (`second`).
  - `vector<pair<int, int>>`: Internal container for the heap.
  - `mycmp`: Custom comparator ensures the heap is ordered by frequency in ascending order.

---

#### **6. Push Frequency Data into the Min-Heap**
```cpp
for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
    pri_que.push(*it);
    if (pri_que.size() > k) {
        pri_que.pop();
    }
}
```
- **Logic**:
  - Iterate through the `map` and push each element (number and frequency) into the min-heap.
  - If the heap size exceeds `k`, remove the top element (the one with the smallest frequency).
- **Effect**: The heap always contains the top `k` elements with the highest frequencies.

---

#### **7. Extract Results from the Min-Heap**
```cpp
vector<int> result(k);
for (int i = k - 1; i >= 0; i--) {
    result[i] = pri_que.top().first;
    pri_que.pop();
}
return result;
```
- Extract the top `k` elements from the heap.
  - `pri_que.top().first`: Gets the number with the smallest frequency.
  - Store the elements in reverse order to ensure the correct result order.
- **Final Output**: Returns the top `k` most frequent elements in the array.

---

### **How the Custom Comparator Works**
1. **Default Behavior**:
   - By default, `priority_queue` in C++ is a **max-heap**, where the largest element is at the top.
2. **Custom Comparator**:
   - The comparator `bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs)` changes this behavior.
   - If `lhs.second > rhs.second`, it means `rhs` has a higher priority and should come earlier.
   - This reverses the default logic, making it behave as a **min-heap**, where the smallest element (by frequency) is at the top.

---

### **Why Does This Create a Min-Heap?**
- The priority queue uses the comparator to decide which element has higher priority.
- In the comparator:
  - Returning `true` means the left element (`lhs`) has **lower priority** than the right element (`rhs`).
  - As a result, the heap ensures that the element with the **smallest frequency** stays at the top.

---

### **Comparison with `sort`**
- In `sort`:
  - If `cmp(a, b)` returns `true`, it means `a` should come **before** `b`.
- In `priority_queue`:
  - If `cmp(a, b)` returns `true`, it means `a` has **lower priority** than `b` and should come **after** `b` in the queue.
- This difference arises because `priority_queue` focuses on maintaining a heap structure where the top element has the highest priority (for a max-heap) or the lowest priority (for a min-heap).

---

### **Example Execution**

**Input**:  
`nums = [1, 1, 1, 2, 2, 3]`, `k = 2`

**Steps**:
1. **Count Frequency**:
   - `map = {1: 3, 2: 2, 3: 1}`.
2. **Build Min-Heap**:
   - Insert `{1, 3}` and `{2, 2}`. Heap size ≤ `k`.
   - Insert `{3, 1}`. Heap size > `k`, so remove `{3, 1}` (smallest frequency).
   - Final heap: `{2, 2}`, `{1, 3}`.
3. **Extract Results**:
   - Extract heap elements in reverse order: `[1, 2]`.

**Output**:  
`[1, 2]`

---

### **Conclusion**
The custom comparator defines the priority logic:
- If `lhs.second > rhs.second`, `rhs` is prioritized.
- This creates a min-heap where the smallest frequency element is at the top.
- This behavior differs from `sort` due to how `priority_queue` interprets the comparator's return value.

This implementation efficiently solves the problem by leveraging the min-heap to maintain the top `k` most frequent elements.