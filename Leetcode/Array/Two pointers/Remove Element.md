[Remove Element](https://leetcode.com/problems/remove-element/description/)

## My intuition solution
```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int len = nums.size();
        vector<int> result;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==val){
                len--;
            }else{
                result.push_back(nums[i]);
            }
        }
        for(int i=0;i<result.size();i++){
            nums[i]=result[i];
        }
        
        return len;
    }
};
```
## Brute force method


```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int size = nums.size();
        for (int i = 0; i < size; i++) {
            if (nums[i] == val) { 
                for (int j = i + 1; j < size; j++) {
                    nums[j - 1] = nums[j];
                }
                i--; 
                size--; 
            }
        }
        return size;

    }
};
```

## Two pointer method

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow =0;
        int fast =0;
        for(fast=0;fast<nums.size();fast++){
            if(nums[fast]!=val){
                nums[slow]=nums[fast];
                slow++;
            }
        }
        return slow;
    }
};
```