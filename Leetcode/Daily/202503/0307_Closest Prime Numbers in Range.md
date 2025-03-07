[Closest Prime Numbers in Range](https://leetcode.com/problems/closest-prime-numbers-in-range/description/)

## **📌 Problem Statement**
Given two positive integers `left` and `right`, find two **prime numbers** `num1` and `num2` such that:

- \( \text{left} \leq \text{num1} < \text{num2} \leq \text{right} \).
- Both `num1` and `num2` are prime numbers.
- \( \text{num2} - \text{num1} \) is the **minimum** among all valid pairs.

Return `[num1, num2]`. If multiple pairs exist, return the one with the **smallest** `num1`. If no valid pair exists, return `[-1, -1]`.

---

## **🔹 Examples**
### **Example 1**
```
Input: left = 10, right = 19
Output: [11,13]
Explanation: The prime numbers in [10,19] are [11, 13, 17, 19].
The closest pair is [11,13] and [17,19], but [11,13] is chosen as num1 is smaller.
```

### **Example 2**
```
Input: left = 4, right = 6
Output: [-1, -1]
Explanation: Only one prime (5) exists in [4,6], so no pair can be formed.
```

---

## **🚀 Approach: Sieve of Eratosthenes**
### **🔑 Idea**
1. Use the **Sieve of Eratosthenes** to efficiently **precompute prime numbers** up to `right`.
2. **Extract prime numbers** in the range `[left, right]`.
3. **Find the closest prime pair** by iterating through the primes and tracking the minimum gap.

---

### **💡 Code (C++)**
```cpp
class Solution {
public:
    vector<int> closestPrimes(int left, int right) {
        vector<bool> sieve(right + 1, true);
        sieve[0] = sieve[1] = false;  // 0 and 1 are not prime
        
        // Step 1: Generate prime numbers using the Sieve of Eratosthenes
        for (int i = 2; i * i <= right; ++i) {
            if (sieve[i]) {
                for (int j = i * i; j <= right; j += i) {
                    sieve[j] = false;
                }
            }
        }
        
        // Step 2: Collect prime numbers in the given range
        vector<int> primes;
        for (int i = left; i <= right; ++i) {
            if (sieve[i]) {
                primes.push_back(i);
            }
        }
        
        // Step 3: Find the closest prime pair
        if (primes.size() < 2) return {-1, -1};
        
        int min_gap = INT_MAX;
        vector<int> result = {-1, -1};
        
        for (int i = 1; i < primes.size(); ++i) {
            int gap = primes[i] - primes[i - 1];
            if (gap < min_gap) {
                min_gap = gap;
                result = {primes[i - 1], primes[i]};
            }
        }
        
        return result;
    }
};
```

---

## **⏳ Complexity Analysis**
- **Sieve of Eratosthenes** runs in **\(O(\text{right} \log \log \text{right})\)**.
- Extracting primes in the range takes **\(O(\text{right} - \text{left})\)**.
- Finding the closest pair takes **\(O(N)\)**, where \(N\) is the number of primes found.

🔹 **Overall Time Complexity**: **\(O(\text{right} \log \log \text{right})\)**  
🔹 **Space Complexity**: **\(O(\text{right})\)** (for the sieve array)

---

## **✅ Summary**
| Approach | Time Complexity | Space Complexity | Notes |
|----------|---------------|----------------|----------------|
| **Sieve of Eratosthenes** | **\(O(\text{right} \log \log \text{right})\)** | **\(O(\text{right})\)** | Efficient for large ranges |

This approach is efficient, leveraging the **Sieve of Eratosthenes** to **precompute primes** and then finding the **closest prime pair** efficiently. 🚀