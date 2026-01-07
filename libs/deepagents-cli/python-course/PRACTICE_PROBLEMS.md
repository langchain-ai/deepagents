# Python Practice Problems - Comprehensive Collection

A curated set of practice problems to strengthen your Python skills. Organized by difficulty level.

## 📋 Table of Contents
- [Beginner Problems](#beginner-problems) (10-15 min each)
- [Intermediate Problems](#intermediate-problems) (20-30 min each)
- [Advanced Problems](#advanced-problems) (30-60 min each)

---

## Beginner Problems

### Problem 1: Sum of Digits
**Difficulty:** ⭐☆☆

Write a function that takes a number and returns the sum of its digits.

```python
def sum_of_digits(n):
    """Return sum of digits in n."""
    # Example: sum_of_digits(123) -> 6
    pass

# Test cases
assert sum_of_digits(123) == 6
assert sum_of_digits(1000) == 1
assert sum_of_digits(999) == 27
```

**Solution:**
```python
def sum_of_digits(n):
    """Return sum of digits in n."""
    return sum(int(digit) for digit in str(abs(n)))
```

---

### Problem 2: Reverse a String
**Difficulty:** ⭐☆☆

Write a function that reverses a string without using slicing.

```python
def reverse_string(s):
    """Reverse string without slicing."""
    # Example: reverse_string("hello") -> "olleh"
    pass

# Test cases
assert reverse_string("hello") == "olleh"
assert reverse_string("a") == "a"
assert reverse_string("12345") == "54321"
```

**Solution:**
```python
def reverse_string(s):
    """Reverse string without slicing."""
    result = ""
    for char in s:
        result = char + result
    return result
```

---

### Problem 3: Check if Palindrome
**Difficulty:** ⭐☆☆

Write a function that checks if a string is a palindrome (ignoring spaces and case).

```python
def is_palindrome(s):
    """Check if string is palindrome."""
    # Example: is_palindrome("A man a plan a canal Panama") -> True
    pass

# Test cases
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("A man a plan a canal Panama") == True
```

**Solution:**
```python
def is_palindrome(s):
    """Check if string is palindrome."""
    clean = s.replace(" ", "").lower()
    return clean == clean[::-1]
```

---

### Problem 4: Count Character Frequency
**Difficulty:** ⭐☆☆

Write a function that counts how many times each character appears in a string.

```python
def char_frequency(s):
    """Count character frequencies."""
    # Example: char_frequency("hello") -> {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    pass

# Test cases
assert char_frequency("hello") == {'h': 1, 'e': 1, 'l': 2, 'o': 1}
assert char_frequency("aaa") == {'a': 3}
```

**Solution:**
```python
def char_frequency(s):
    """Count character frequencies."""
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq
```

---

### Problem 5: List Operations
**Difficulty:** ⭐☆☆

Write functions for common list operations.

```python
def find_max(lst):
    """Find maximum without using max()."""
    pass

def find_min(lst):
    """Find minimum without using min()."""
    pass

def remove_duplicates(lst):
    """Remove duplicates, preserve order."""
    pass

# Test cases
assert find_max([3, 1, 4, 1, 5]) == 5
assert find_min([3, 1, 4, 1, 5]) == 1
assert remove_duplicates([1, 2, 2, 3, 3, 3]) == [1, 2, 3]
```

**Solution:**
```python
def find_max(lst):
    max_val = lst[0]
    for num in lst:
        if num > max_val:
            max_val = num
    return max_val

def find_min(lst):
    min_val = lst[0]
    for num in lst:
        if num < min_val:
            min_val = num
    return min_val

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
```

---

## Intermediate Problems

### Problem 6: Fibonacci Sequence
**Difficulty:** ⭐⭐☆

Write a function that generates Fibonacci numbers efficiently.

```python
def fibonacci(n):
    """Generate first n Fibonacci numbers."""
    # Example: fibonacci(5) -> [0, 1, 1, 2, 3]
    pass

# Test cases
assert fibonacci(1) == [0]
assert fibonacci(5) == [0, 1, 1, 2, 3]
assert fibonacci(7) == [0, 1, 1, 2, 3, 5, 8]
```

**Solution:**
```python
def fibonacci(n):
    """Generate first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]
```

---

### Problem 7: Prime Numbers
**Difficulty:** ⭐⭐☆

Write a function to find all prime numbers up to n.

```python
def sieve_of_eratosthenes(n):
    """Find all primes up to n."""
    # Example: sieve_of_eratosthenes(10) -> [2, 3, 5, 7]
    pass

# Test cases
assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]
assert sieve_of_eratosthenes(20) == [2, 3, 5, 7, 11, 13, 17, 19]
```

**Solution:**
```python
def sieve_of_eratosthenes(n):
    """Find all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]
```

---

### Problem 8: Two Sum Problem
**Difficulty:** ⭐⭐☆

Given a list and a target sum, find two numbers that add up to the target.

```python
def two_sum(nums, target):
    """Find two numbers that sum to target."""
    # Example: two_sum([2, 7, 11, 15], 9) -> (2, 7)
    # Returns None if not found
    pass

# Test cases
assert two_sum([2, 7, 11, 15], 9) == (2, 7)
assert two_sum([3, 2, 4], 6) == (2, 4)
assert two_sum([1, 2, 3], 10) is None
```

**Solution:**
```python
def two_sum(nums, target):
    """Find two numbers that sum to target."""
    seen = {}
    for num in nums:
        complement = target - num
        if complement in seen:
            return (complement, num)
        seen[num] = True
    return None
```

---

### Problem 9: Word Frequency
**Difficulty:** ⭐⭐☆

Find the most frequent word in a text.

```python
def most_frequent_word(text):
    """Find the most frequent word."""
    # Example: most_frequent_word("the quick brown fox jumps") 
    #          -> "the" (assuming case-insensitive)
    pass

# Test cases
result = most_frequent_word("apple banana apple cherry apple")
assert result[0] == "apple"  # Word
assert result[1] == 3        # Frequency
```

**Solution:**
```python
def most_frequent_word(text):
    """Find the most frequent word."""
    words = text.lower().split()
    freq = {}
    
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    
    most_common = max(freq.items(), key=lambda x: x[1])
    return most_common
```

---

### Problem 10: Matrix Transposition
**Difficulty:** ⭐⭐☆

Write a function to transpose a matrix.

```python
def transpose_matrix(matrix):
    """Transpose a matrix."""
    # Example: transpose_matrix([[1, 2], [3, 4]])
    #          -> [[1, 3], [2, 4]]
    pass

# Test cases
assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
assert transpose_matrix([[1, 2, 3]]) == [[1], [2], [3]]
```

**Solution:**
```python
def transpose_matrix(matrix):
    """Transpose a matrix."""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
```

---

## Advanced Problems

### Problem 11: Longest Substring Without Repeating
**Difficulty:** ⭐⭐⭐

Find the longest substring without repeating characters.

```python
def longest_substring(s):
    """Find longest substring without repeating chars."""
    # Example: longest_substring("abcabcbb") -> "abc" (length 3)
    # Returns the substring, not just length
    pass

# Test cases
assert longest_substring("abcabcbb") == "abc"
assert longest_substring("bbbbb") == "b"
assert longest_substring("pwwkew") == "wke"
```

**Solution:**
```python
def longest_substring(s):
    """Find longest substring without repeating chars."""
    char_index = {}
    max_length = 0
    max_start = 0
    start = 0
    
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = i
        current_length = i - start + 1
        
        if current_length > max_length:
            max_length = current_length
            max_start = start
    
    return s[max_start:max_start + max_length]
```

---

### Problem 12: Merge Sorted Lists
**Difficulty:** ⭐⭐⭐

Merge two sorted lists into one sorted list efficiently.

```python
def merge_sorted_lists(list1, list2):
    """Merge two sorted lists."""
    # Example: merge_sorted_lists([1, 3, 5], [2, 4, 6])
    #          -> [1, 2, 3, 4, 5, 6]
    pass

# Test cases
assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted_lists([1], [1]) == [1, 1]
assert merge_sorted_lists([], [1, 2]) == [1, 2]
```

**Solution:**
```python
def merge_sorted_lists(list1, list2):
    """Merge two sorted lists."""
    result = []
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result
```

---

### Problem 13: Binary Search
**Difficulty:** ⭐⭐⭐

Implement binary search on a sorted list.

```python
def binary_search(lst, target):
    """Find target in sorted list using binary search."""
    # Returns index if found, -1 if not found
    pass

# Test cases
assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 6) == -1
assert binary_search([1], 1) == 0
```

**Solution:**
```python
def binary_search(lst, target):
    """Find target in sorted list using binary search."""
    left, right = 0, len(lst) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

---

### Problem 14: Valid Parentheses
**Difficulty:** ⭐⭐⭐

Check if parentheses/brackets are properly matched.

```python
def is_valid_parentheses(s):
    """Check if parentheses are balanced."""
    # Example: is_valid_parentheses("()[]{}") -> True
    # Example: is_valid_parentheses("([)]") -> False
    pass

# Test cases
assert is_valid_parentheses("()") == True
assert is_valid_parentheses("()[]{}") == True
assert is_valid_parentheses("([)]") == False
assert is_valid_parentheses("{[}]") == False
```

**Solution:**
```python
def is_valid_parentheses(s):
    """Check if parentheses are balanced."""
    stack = []
    matching = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in matching:
            stack.append(char)
        else:
            if not stack or matching[stack.pop()] != char:
                return False
    
    return len(stack) == 0
```

---

### Problem 15: Group Anagrams
**Difficulty:** ⭐⭐⭐

Group words that are anagrams of each other.

```python
def group_anagrams(words):
    """Group anagrams together."""
    # Example: group_anagrams(["eat", "tea", "ate", "tan", "nat", "ant"])
    #          -> [["eat", "tea", "ate"], ["tan", "nat", "ant"]]
    pass

# Test cases
result = group_anagrams(["eat", "tea", "ate", "tan", "nat", "ant"])
assert sorted([sorted(g) for g in result]) == \
       [["ant", "nat", "tan"], ["ate", "eat", "tea"]]
```

**Solution:**
```python
def group_anagrams(words):
    """Group anagrams together."""
    anagram_map = {}
    
    for word in words:
        # Sort letters to use as key
        sorted_word = ''.join(sorted(word))
        if sorted_word not in anagram_map:
            anagram_map[sorted_word] = []
        anagram_map[sorted_word].append(word)
    
    return list(anagram_map.values())
```

---

## Tips for Problem Solving

### Approach
1. **Understand the problem** - Read carefully, clarify edge cases
2. **Think of examples** - Write out test cases
3. **Plan your solution** - Use pseudocode first
4. **Implement** - Write clean code
5. **Test** - Verify with multiple cases
6. **Optimize** - Look for improvements

### Common Patterns
- **Sliding window** - For substring problems
- **Two pointers** - For sorted arrays
- **Hash map** - For frequency/lookup problems
- **Stack/queue** - For bracket/order problems
- **Recursion** - For tree/graph problems

### Optimization
- Use **sets** for O(1) lookup instead of lists
- Use **dictionaries** for frequency counting
- Consider **space vs time** tradeoffs
- Think about **worst case** complexity

---

## Difficulty Progression

```
Beginner        ⭐☆☆
- Simple loops
- String operations
- List manipulation
- Basic dictionaries

Intermediate    ⭐⭐☆
- Efficient algorithms
- Data structure selection
- Problem-solving patterns
- Complexity analysis

Advanced        ⭐⭐⭐
- Complex algorithms
- Optimization techniques
- Multiple data structures
- Edge case handling
```

---

## Resources

### Online Judge Platforms
- LeetCode - leetcode.com
- HackerRank - hackerrank.com
- CodeWars - codewars.com
- Project Euler - projecteuler.net

### Books
- "Cracking the Coding Interview" by Gayle Laakmann McDowell
- "Introduction to Algorithms" (CLRS)

### Practice Strategy
1. **Daily practice** - 30-60 minutes
2. **Mix difficulties** - Don't just do easy problems
3. **Review solutions** - Learn better approaches
4. **Explain to others** - Teach what you learned
5. **Track progress** - Know what you've mastered

---

## Solutions Organization

All solutions follow these principles:
- ✓ Clear variable names
- ✓ Comments for complex logic
- ✓ Proper error handling
- ✓ Efficient algorithms
- ✓ Clean formatting

---

**Remember:** The goal is understanding, not just solving. If you get stuck:
1. Review the hints
2. Think about the algorithm
3. Look at the solution
4. Implement it yourself
5. Solve similar problems

Good luck with your practice! 💪

---

Last Updated: 2024
Python Version: 3.8+
