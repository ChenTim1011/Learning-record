[Design a Number Container System](https://leetcode.com/problems/design-a-number-container-system/description/)

### **Problem: Design a Number Container System**

1. **Insert or Replace a Number at a Given Index:**
   - Assign a number to a specific index in the system. If the index already has a number, replace it with the new one.
   
2. **Return the Smallest Index for a Given Number:**
   - Query the system for the smallest index that currently stores the given number. If no such index exists, return `-1`.

---

### **Class Definition**
You need to implement the `NumberContainers` class:

- **`NumberContainers()`**: Initializes the system.
- **`void change(int index, int number)`**:
  - Fills the container at `index` with `number`. If `index` already has a number, replace it with the new value.
- **`int find(int number)`**:
  - Returns the smallest index that stores `number`. If `number` does not exist in the system, return `-1`.

---

### **Example**

#### Input:
```plaintext
["NumberContainers", "find", "change", "change", "change", "change", "find", "change", "find"]
[[], [10], [2, 10], [1, 10], [3, 10], [5, 10], [10], [1, 20], [10]]
```

#### Output:
```plaintext
[null, -1, null, null, null, null, 1, null, 2]
```

#### Explanation:
1. `NumberContainers nc = new NumberContainers();`
   - Initializes an empty system.

2. `nc.find(10);`
   - There are no indices containing `10`, so return `-1`.

3. `nc.change(2, 10);`
   - Set index `2` to `10`.

4. `nc.change(1, 10);`
   - Set index `1` to `10`.

5. `nc.change(3, 10);`
   - Set index `3` to `10`.

6. `nc.change(5, 10);`
   - Set index `5` to `10`.

7. `nc.find(10);`
   - Indices containing `10` are `[1, 2, 3, 5]`. The smallest index is `1`, so return `1`.

8. `nc.change(1, 20);`
   - Replace the number at index `1` with `20`. Now `10` is only at indices `[2, 3, 5]`.

9. `nc.find(10);`
   - The smallest index containing `10` is `2`, so return `2`.

---

### **Constraints**
1. \( 1 \leq \text{index}, \text{number} \leq 10^9 \)
2. At most \( 10^5 \) calls will be made in total to `change` and `find`.

---

### **Solution Explanation**

To efficiently implement the system, we can use two hash maps:

1. **`mp`: `unordered_map<int, int>`**
   - Maps an `index` to its current `number`. 
   - This helps us track the number currently stored at each index.

2. **`idx`: `unordered_map<int, set<int>>`**
   - Maps a `number` to a set of indices that currently store this number.
   - Using a `set` ensures that the indices are stored in sorted order, which allows us to quickly retrieve the smallest index.

---

### **Algorithm**

#### **1. `change(int index, int number)`**
1. Check if `index` is already present in `mp`:
   - If it exists, get the old number associated with this index.
   - Remove the `index` from the set of indices in `idx[old_number]`.
   - If `idx[old_number]` becomes empty, remove the entry for `old_number` from `idx`.
2. Update `mp[index]` to the new `number`.
3. Add `index` to the set of indices in `idx[number]`.

#### **2. `find(int number)`**
1. Check if `number` exists in `idx`:
   - If it does, return the smallest element in `idx[number]` (i.e., the first element in the set).
   - If it doesn't exist, return `-1`.

---

### **Code Implementation (C++)**

```cpp
#include <unordered_map>
#include <set>
using namespace std;

class NumberContainers {
public:
    unordered_map<int, int> mp; // index -> number
    unordered_map<int, set<int>> idx; // number -> set of indices

    NumberContainers() {
        mp.reserve(100000); // Reserve space to reduce rehashing
    }
    
    void change(int index, int number) {
        // Check if the index is already in mp
        if (mp.count(index)) {
            int oldNumber = mp[index]; // Get the old number
            idx[oldNumber].erase(index); // Remove index from old number's set
            if (idx[oldNumber].empty()) idx.erase(oldNumber); // Remove entry if set is empty
        }
        // Update mp and idx with the new number
        mp[index] = number;
        idx[number].insert(index);
    }
    
    int find(int number) {
        // If number exists in idx, return the smallest index
        if (idx.count(number) == 0) return -1;
        return *(idx[number].begin());
    }
};
```

---

### **Step-by-Step Example**

#### **Input Operations:**
```plaintext
["NumberContainers", "find", "change", "change", "find"]
[[], [10], [2, 10], [3, 10], [10]]
```

1. **`NumberContainers`**: Initialize `mp = {}` and `idx = {}`.
2. **`find(10)`**: Return `-1` because `10` is not in the system.
3. **`change(2, 10)`**:
   - Update `mp = {2: 10}` and `idx = {10: {2}}`.
4. **`change(3, 10)`**:
   - Update `mp = {2: 10, 3: 10}` and `idx = {10: {2, 3}}`.
5. **`find(10)`**: Return `2` because the smallest index for `10` is `2`.

#### **Output:**
```plaintext
[null, -1, null, null, 2]
```

---

### **Complexity Analysis**

1. **`change()`**:
   - Removing an index from a set: \( O(\log n) \)
   - Adding an index to a set: \( O(\log n) \)
   - Overall: \( O(\log n) \)

2. **`find()`**:
   - Accessing the smallest index from a set: \( O(1) \)
   - Overall: \( O(1) \)

3. **Space Complexity**:
   - `mp` stores up to \( O(k) \) entries, where \( k \) is the number of unique indices.
   - `idx` stores sets for each unique number, with a total of up to \( O(k) \) entries.

Total space and time complexities are efficient for the constraint \( 10^5 \) operations.