[Longest Nice Subarray](https://leetcode.com/problems/longest-nice-subarray/description/?envType=daily-question&envId=2025-03-18)

## **📌 Problem Statement**  
We are given an **array `nums`** consisting of **positive integers**.  

A **subarray** of `nums` is called **nice** if:  
- The **bitwise AND** (`&`) of **every pair of elements** in the subarray is **0** (i.e., no two elements share the same bit in their binary representation).  

Our goal is to **return the length of the longest nice subarray**.  

🔹 **Key Observations:**  
- **Subarrays are contiguous**, meaning we cannot rearrange elements.  
- **Bitwise AND (`&`)** will be `0` **only if there are no overlapping bits** between any two numbers in the subarray.  
- Any **single element** is a **valid nice subarray** since `x & x = x` and it has no other elements to compare.

---

## **🔹 Example Walkthrough**  

### **Example 1**  
#### **Input:**  
```cpp
nums = [1,3,8,48,10]
```
#### **Binary Representation:**  
| Number | Binary |  
|---------|----------------|  
| `1`     | `00001`        |  
| `3`     | `00011`        |  
| `8`     | `01000`        |  
| `48`    | `110000`       |  
| `10`    | `01010`        |  

#### **Process:**  
We want the **longest subarray** such that no two numbers share a `1` in the same bit position.  

- **Checking subarray `[3, 8, 48]`**  
  - `3 & 8 = 00011 & 01000 = 00000` ✅  
  - `3 & 48 = 00011 & 110000 = 00000` ✅  
  - `8 & 48 = 01000 & 110000 = 00000` ✅  
- This is the **longest** valid subarray → **Length = 3**  

#### **Output:**  
```cpp
3
```

---

### **Example 2**  
#### **Input:**  
```cpp
nums = [3,1,5,11,13]
```
#### **Binary Representation:**  
| Number | Binary |  
|---------|----------------|  
| `3`     | `00011`        |  
| `1`     | `00001`        |  
| `5`     | `00101`        |  
| `11`    | `01011`        |  
| `13`    | `01101`        |  

#### **Process:**  
- **Any two numbers have overlapping bits**, meaning no subarray longer than **1** exists.  
- The answer is **1**.  

#### **Output:**  
```cpp
1
```

---

## **🔹 Approach: Sliding Window with Bitmask**
### **💡 Idea:**  
We use a **sliding window** (`left` to `right`) while maintaining a **bitmask** of the current subarray.  
1. Expand the window (`right++`) while maintaining the **bitwise AND (`&`) = 0** property.  
2. If adding `nums[right]` causes overlap (i.e., `bitmask & nums[right] ≠ 0`), **move `left` forward** until we restore a valid subarray.  
3. Track the **maximum window length**.  

### **🔹 Implementation**
```cpp
class Solution {
public:
    int longestNiceSubarray(vector<int>& nums) {
        int left = 0, bitmask = 0, maxLength = 0;
        
        for (int right = 0; right < nums.size(); right++) {
            // Ensure new element doesn't overlap with existing bitmask
            while ((bitmask & nums[right]) != 0) {
                bitmask ^= nums[left]; // Remove left element
                left++;
            }
            
            // Add new element to the bitmask
            bitmask |= nums[right];
            
            // Update max length
            maxLength = max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
};
```

---

## **🔹 Complexity Analysis**
| Approach | Time Complexity | Space Complexity | Explanation |
|----------|---------------|----------------|-------------|
| **Sliding Window + Bitmask** | **O(n)** | **O(1)** | Each element is added & removed **at most once**. |

---

## **🔹 Summary**
✅ **Key Idea**: Use a **bitmask + sliding window** to keep track of valid subarrays.  
✅ **Efficient**: O(n) time complexity since each element is processed at most once.  
✅ **Best Approach**: Sliding window with **bitwise operations** optimally maintains the **longest nice subarray**.  

