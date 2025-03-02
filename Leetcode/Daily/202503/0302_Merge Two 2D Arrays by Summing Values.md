[Merge Two 2D Arrays by Summing Values](https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values/description/)


## **📌 Problem Statement**  
You are given two **2D integer arrays** `nums1` and `nums2`.  

- `nums1[i] = [idi, vali]` indicates that the number with **id = idi** has a value of `vali`.  
- `nums2[i] = [idi, vali]` represents the same meaning.  

Each array is **sorted in strictly increasing order** based on `id`, and each `id` is **unique** in its respective array.  

Return a merged array where:  
1. Each `id` appears **only once**, and its value is the **sum** of the values from both arrays.  
2. If an `id` appears in only one array, assume the value from the other array is `0`.  
3. The result should be **sorted in ascending order** by `id`.  

---

## **Example Walkthrough**  

### **Example 1**  
#### **Input:**  
```cpp
nums1 = [[1,2],[2,3],[4,5]];
nums2 = [[1,4],[3,2],[4,1]];
```
#### **Explanation:**  
- `1` appears in both arrays: `2 + 4 = 6`  
- `2` appears only in `nums1`: `3`  
- `3` appears only in `nums2`: `2`  
- `4` appears in both arrays: `5 + 1 = 6`  

#### **Output:**  
```cpp
[[1,6],[2,3],[3,2],[4,6]]
```

---

### **Example 2**  
#### **Input:**  
```cpp
nums1 = [[2,4],[3,6],[5,5]];
nums2 = [[1,3],[4,3]];
```
#### **Explanation:**  
- `1` appears only in `nums2`: `3`  
- `2` appears only in `nums1`: `4`  
- `3` appears only in `nums1`: `6`  
- `4` appears only in `nums2`: `3`  
- `5` appears only in `nums1`: `5`  

#### **Output:**  
```cpp
[[1,3],[2,4],[3,6],[4,3],[5,5]]
```

---

## **Optimized Approach**  
### **Key Observations**  
1. Both `nums1` and `nums2` are **sorted** by `id`.  
2. We can efficiently **merge** them using the **two-pointer technique** in **O(n + m)** time.  

---

## **Optimized C++ Solution**  
```cpp
class Solution {
public:
    vector<vector<int>> mergeArrays(vector<vector<int>>& nums1, vector<vector<int>>& nums2) {
        vector<vector<int>> result;
        int left = 0, right = 0;

        while (left < nums1.size() && right < nums2.size()) {
            if (nums1[left][0] == nums2[right][0]) {
                result.push_back({nums1[left][0], nums1[left][1] + nums2[right][1]});
                left++;
                right++;
            } else if (nums1[left][0] < nums2[right][0]) {
                result.push_back(nums1[left++]);
            } else {
                result.push_back(nums2[right++]);
            }
        }

        while (left < nums1.size()) {
            result.push_back(nums1[left++]);
        }

        while (right < nums2.size()) {
            result.push_back(nums2[right++]);
        }

        return result;
    }
};
```

---

## **Explanation of Code**  

### **1️⃣ Initialize Two Pointers**
```cpp
int left = 0, right = 0;
```
- `left` tracks position in `nums1`, `right` tracks position in `nums2`.  

---

### **2️⃣ Merge Two Sorted Arrays Using Two Pointers**
```cpp
while (left < nums1.size() && right < nums2.size()) {
```
- **Case 1**: Same `id` → Sum values and move both pointers.  
```cpp
if (nums1[left][0] == nums2[right][0]) {
    result.push_back({nums1[left][0], nums1[left][1] + nums2[right][1]});
    left++;
    right++;
}
```
- **Case 2**: `nums1` has a smaller `id` → Add it to `result` and move `left`.  
```cpp
else if (nums1[left][0] < nums2[right][0]) {
    result.push_back(nums1[left++]);
}
```
- **Case 3**: `nums2` has a smaller `id` → Add it to `result` and move `right`.  
```cpp
else {
    result.push_back(nums2[right++]);
}
```

---

### **3️⃣ Add Remaining Elements**
```cpp
while (left < nums1.size()) {
    result.push_back(nums1[left++]);
}
while (right < nums2.size()) {
    result.push_back(nums2[right++]);
}
```
- If one array is fully processed while the other still has elements, append them directly.  

---

## **Complexity Analysis**  
| Complexity | Explanation |
|------------|------------|
| **Time Complexity** | **O(n + m)** → Each element from `nums1` and `nums2` is processed once. |
| **Space Complexity** | **O(n + m)** → Output array stores all elements. |

---

## **Edge Cases Considered**  
✅ **No common `id`s:**  
   - Example: `nums1 = [[1,2],[2,3]]`, `nums2 = [[3,4],[4,5]]`  
   - Output: `[[1,2],[2,3],[3,4],[4,5]]`  

✅ **All `id`s are common:**  
   - Example: `nums1 = [[1,2],[2,3]]`, `nums2 = [[1,4],[2,6]]`  
   - Output: `[[1,6],[2,9]]`  

✅ **One array is empty:**  
   - Example: `nums1 = []`, `nums2 = [[1,2],[2,3]]`  
   - Output: `[[1,2],[2,3]]`  

✅ **Handling large input sizes efficiently:**  
   - Works efficiently for `nums1.size() = 200`, `nums2.size() = 200`.  

---

## **Summary**  
✅ **Uses `O(n + m)` efficient two-pointer merging**  
✅ **Maintains sorted order without extra sorting**  
✅ **Handles all edge cases properly**  
✅ **Simple, clean, and easy-to-understand implementation** 🚀