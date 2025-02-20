[Find Unique Binary String](https://leetcode.com/problems/find-unique-binary-string/description/)


## **Approach Explanation**
Instead of generating all possible binary strings of length `n` and checking for a missing one (which would take exponential time **O(2ⁿ)**), we **construct a unique string directly** using a **diagonal flipping technique**.

### **Key Idea: Diagonal Flipping**
1. We observe that the given input `nums` contains `n` unique binary strings, each of length `n`.
2. To ensure that the new string differs from every string in `nums`, we construct it **by flipping the diagonal elements** (i.e., the `i-th` character from the `i-th` string).
3. This guarantees that the new string is different from each `nums[i]` at **least one position**—specifically, the `i-th` position.

### **Why Does This Work?**
- If the newly constructed string **were already in `nums`**, then it must be equal to some `nums[i]`.
- However, by construction, our new string differs from `nums[i]` at the `i-th` character.
- This contradiction means the newly formed string **cannot be present in `nums`**, making it a valid answer.

---

## **Code Implementation**
```cpp
class Solution {
public:
    string findDifferentBinaryString(vector<string>& nums) {
        for (int i = 0; i < nums.size(); i++)
            nums[0][i] = nums[i][i] == '1' ? '0' : '1';
        return nums[0];
    }
};
```

---

## **Code Explanation**
1. **Loop Through `nums`**:
   - We iterate over indices `i = 0` to `n-1`.
   - For each index `i`, we check the `i-th` character of `nums[i]` (i.e., `nums[i][i]`).
   - If `nums[i][i]` is `'1'`, we set it to `'0'`; otherwise, we set it to `'1'`.
  
2. **Storing the Result in `nums[0]`**:
   - Instead of using extra space, we modify `nums[0]` directly to store our answer (this is only for languages like C++ that allow string modification).
  
3. **Return the Modified `nums[0]`**:
   - Since the constructed string is guaranteed to be unique, we return it.

---

## **Time and Space Complexity Analysis**
- **Time Complexity**: **O(n)**  
  - We iterate through `n` elements exactly once, performing constant-time operations.
- **Space Complexity**:
  - **O(1)** if we modify `nums[0]` directly (saving space).
  - **O(n)** if we create a new string to store the result.

---

## **Example Walkthrough**
### **Example 1**
```cpp
Input: nums = ["01", "10"]
```
| i | nums[i] | nums[i][i] | Flipped |
|---|--------|-----------|---------|
| 0 | "01"   | '0'       | '1'     |
| 1 | "10"   | '0'       | '1'     |

Generated string: **"11"**  
**Output:** `"11"` (Other valid answers: `"00"`)

---

### **Example 2**
```cpp
Input: nums = ["00", "01"]
```
| i | nums[i] | nums[i][i] | Flipped |
|---|--------|-----------|---------|
| 0 | "00"   | '0'       | '1'     |
| 1 | "01"   | '1'       | '0'     |

Generated string: **"10"**  
**Output:** `"10"` (Other valid answers: `"11"`)

---

### **Example 3**
```cpp
Input: nums = ["111", "011", "001"]
```
| i | nums[i] | nums[i][i] | Flipped |
|---|--------|-----------|---------|
| 0 | "111"  | '1'       | '0'     |
| 1 | "011"  | '1'       | '0'     |
| 2 | "001"  | '1'       | '0'     |

Generated string: **"000"**  
**Output:** `"000"` (Other valid answers: `"010"`, `"100"`, `"110"`)

---

## **Why This Approach is Efficient?**
1. **Avoids Exponential Search**:  
   - Instead of checking all **O(2ⁿ)** possible strings, we construct the missing string in **O(n)** time.
  
2. **Guaranteed to be Unique**:  
   - The diagonal flipping ensures the result differs from each `nums[i]` at the `i-th` position.
  
3. **Simple to Implement**:  
   - Just a single loop and direct character manipulation.

---

## **Conclusion**
- **Key technique**: **Diagonal flipping**
- **Time complexity**: **O(n)**
- **Space complexity**: **O(1) (if modifying `nums[0]`), otherwise O(n)**
- **Constructs a guaranteed missing binary string efficiently**.

This approach is an elegant and efficient way to find a unique binary string in **linear time**. 🚀