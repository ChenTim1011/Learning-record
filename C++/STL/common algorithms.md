### **Common STL Algorithms in C++**

The Standard Template Library (STL) in C++ provides a rich set of algorithms for performing various operations on data structures. These algorithms are classified into several categories based on their purpose, as outlined below. Each category is introduced with examples and explanations.

---

### **1. Non-Modifying Sequence Algorithms**

These algorithms operate on a sequence of elements but do not modify the elements in the sequence.

#### Common Algorithms:
- `std::for_each`
- `std::count`
- `std::find`
- `std::binary_search`

#### **Examples**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 2};

    // 1. std::for_each: Apply a function to each element
    std::cout << "Elements: ";
    std::for_each(nums.begin(), nums.end(), [](int n) {
        std::cout << n << " ";
    });
    std::cout << std::endl;

    // 2. std::count: Count occurrences of an element
    int count_of_2 = std::count(nums.begin(), nums.end(), 2);
    std::cout << "Count of 2: " << count_of_2 << std::endl;

    // 3. std::find: Find the first occurrence of an element
    auto it = std::find(nums.begin(), nums.end(), 3);
    if (it != nums.end()) {
        std::cout << "Found 3 at position: " << std::distance(nums.begin(), it) << std::endl;
    }

    // 4. std::binary_search: Check if an element exists (requires sorted data)
    std::sort(nums.begin(), nums.end());
    bool exists = std::binary_search(nums.begin(), nums.end(), 4);
    std::cout << "4 exists: " << std::boolalpha << exists << std::endl;

    return 0;
}
```

---

### **2. Modifying Sequence Algorithms**

These algorithms modify the elements of a sequence.

#### Common Algorithms:
- `std::sort`
- `std::reverse`
- `std::swap`
- `std::rotate`

#### **Examples**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {3, 1, 4, 1, 5};

    // 1. std::sort: Sort elements in ascending order
    std::sort(nums.begin(), nums.end());
    std::cout << "Sorted: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    // 2. std::reverse: Reverse the order of elements
    std::reverse(nums.begin(), nums.end());
    std::cout << "Reversed: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    // 3. std::swap: Swap two elements
    std::swap(nums[0], nums[1]);
    std::cout << "After swap: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    // 4. std::rotate: Rotate elements
    std::rotate(nums.begin(), nums.begin() + 2, nums.end());
    std::cout << "After rotate: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
```

---

### **3. Permutation and Combination Algorithms**

These algorithms are used to generate permutations or merge sequences.

#### Common Algorithms:
- `std::next_permutation`
- `std::prev_permutation`
- `std::merge`

#### **Examples**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {1, 2, 3};

    // 1. std::next_permutation: Generate the next lexicographical permutation
    std::cout << "Permutations:\n";
    do {
        for (int n : nums) std::cout << n << " ";
        std::cout << std::endl;
    } while (std::next_permutation(nums.begin(), nums.end()));

    // 2. std::merge: Merge two sorted ranges
    std::vector<int> nums1 = {1, 3, 5}, nums2 = {2, 4, 6}, merged(6);
    std::merge(nums1.begin(), nums1.end(), nums2.begin(), nums2.end(), merged.begin());
    std::cout << "Merged: ";
    for (int n : merged) std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
```

---

### **4. Numerical Algorithms**

These algorithms perform numerical computations on sequences.

#### Common Algorithms:
- `std::accumulate`
- `std::inner_product`
- `std::partial_sum`

#### **Examples**:
```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> nums = {1, 2, 3, 4};

    // 1. std::accumulate: Calculate the sum of elements
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    std::cout << "Sum: " << sum << std::endl;

    // 2. std::inner_product: Compute inner product of two ranges
    std::vector<int> nums2 = {5, 6, 7, 8};
    int dot_product = std::inner_product(nums.begin(), nums.end(), nums2.begin(), 0);
    std::cout << "Dot Product: " << dot_product << std::endl;

    // 3. std::partial_sum: Compute partial sums
    std::vector<int> partial(nums.size());
    std::partial_sum(nums.begin(), nums.end(), partial.begin());
    std::cout << "Partial sums: ";
    for (int n : partial) std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
```

---

### **5. Heap Operations**

These algorithms allow you to manipulate a heap.

#### Common Algorithms:
- `std::make_heap`
- `std::push_heap`
- `std::pop_heap`

#### **Examples**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {4, 1, 3, 2, 5};

    // 1. std::make_heap: Turn a range into a max-heap
    std::make_heap(nums.begin(), nums.end());
    std::cout << "Heap: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    // 2. std::pop_heap: Remove the top element (largest in max-heap)
    std::pop_heap(nums.begin(), nums.end());
    nums.pop_back();  // Actually remove the element
    std::cout << "After pop: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    // 3. std::push_heap: Add a new element to the heap
    nums.push_back(6);
    std::push_heap(nums.begin(), nums.end());
    std::cout << "After push: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
```

---

### **6. Partition Algorithms**

These algorithms divide a sequence into two parts based on a condition.

#### Common Algorithms:
- `std::partition`
- `std::stable_partition`

#### **Examples**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6};

    // 1. std::partition: Partition elements based on a condition
    std::partition(nums.begin(), nums.end(), [](int n) { return n % 2 == 0; });
    std::cout << "Partitioned: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
```

---

### **Conclusion**

The STL algorithm library in C++ is extensive and powerful, covering almost all common operations on sequences, heaps, and numerical computations. Mastering these algorithms will help you write efficient and concise code for a wide range of applications.