[Divide Array Into Equal Pairs](https://leetcode.com/problems/divide-array-into-equal-pairs/description/?envType=daily-question&envId=2025-03-17)

## **📌 Problem Statement**  
We are given an **integer array** `nums` of length **2 * n** (i.e., the length is always even).  

We need to **divide `nums` into `n` pairs**, ensuring:  
1. **Each element belongs to exactly one pair.**  
2. **Each pair contains two equal elements.**  

We must **return `true` if it is possible** to divide `nums` into `n` pairs, otherwise **return `false`**.  

---

## **🔹 Example Walkthrough**  

### **Example 1**  
#### **Input:**  
```cpp
nums = [3,2,3,2,2,2]
```
#### **Process:**  
- The length is **6**, so we must form **3 pairs**.  
- We can create pairs: **(2,2), (3,3), (2,2)**.  
- All conditions are satisfied ✅.  

#### **Output:**  
```cpp
true
```

---

### **Example 2**  
#### **Input:**  
```cpp
nums = [1,2,3,4]
```
#### **Process:**  
- The length is **4**, so we must form **2 pairs**.  
- However, **there are no two equal numbers**, so it is impossible ❌.  

#### **Output:**  
```cpp
false
```

---

## **🔹 Key Observations**  
1. Since the length is always **even**, the only way to divide `nums` into pairs is if **every number appears an even number of times**.  
2. If any number appears **an odd number of times**, it means that one element **cannot be paired**, so we return `false`.  

---

## **🔹 Approach 1: Using HashMap (Frequency Count)**
### **💡 Idea:**  
1. **Count the frequency** of each number.  
2. **Check if every frequency is even**.  
   - If **all numbers appear an even number of times**, return `true`.  
   - Otherwise, return `false`.  

### **🔹 Implementation – Using `unordered_map`**
```cpp
class Solution {
public:
    bool divideArray(vector<int>& nums) {
        unordered_map<int, int> freq;
        
        // Count the frequency of each number
        for (int num : nums) {
            freq[num]++;
        }
        
        // Check if all frequencies are even
        for (auto& [key, count] : freq) {
            if (count % 2 != 0) {
                return false;
            }
        }
        return true;
    }
};
```

---

## **🔹 Approach 2: Using Sorting**
### **💡 Idea:**  
1. **Sort the array**.  
2. **Check adjacent elements**:  
   - Since identical numbers will be next to each other after sorting, we can easily check them in **pairs**.  
   - If `nums[i] != nums[i+1]` at any point, return `false`.  

### **🔹 Implementation – Using Sorting**
```cpp
class Solution {
public:
    bool divideArray(vector<int>& nums) {
        sort(nums.begin(), nums.end()); // Sort the array
        
        // Check pairs
        for (int i = 0; i < nums.size(); i += 2) {
            if (nums[i] != nums[i + 1]) {
                return false; // If a pair is not equal, return false
            }
        }
        return true;
    }
};
```

---

## **🔹 Complexity Analysis**
| Approach | Time Complexity | Space Complexity | Explanation |
|----------|---------------|----------------|-------------|
| **HashMap (Frequency Count)** | **O(n)** | **O(n)** | Count frequency in O(n), check in O(n). |
| **Sorting** | **O(n log n)** | **O(1)** | Sorting takes O(n log n), checking pairs takes O(n). |

**Best Approach:**  
- **If `nums` is small (`n ≤ 500`)**, sorting is **simple and efficient**.  
- **For large `n`**, **hashmap** is **optimal** with O(n) time.  

---

## **🔹 Summary**
✅ **Key Idea**: Every number must appear **an even number of times** to form pairs.  
✅ **Two Approaches**:  
1. **Using HashMap** – Count frequency and check if all counts are even.  
2. **Using Sorting** – Sort and check adjacent pairs.  
✅ **Time Complexity**:  
- **O(n)** (HashMap) → Best for large inputs.  
- **O(n log n)** (Sorting) → Simple and works well for small inputs.  
