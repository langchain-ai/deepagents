# Python Interview Preparation Guide

Master the key concepts and problems for Python interviews.

## 📋 Table of Contents
1. [Pre-Interview Checklist](#pre-interview-checklist)
2. [Core Concepts Questions](#core-concepts-questions)
3. [Coding Interview Patterns](#coding-interview-patterns)
4. [Common Interview Problems](#common-interview-problems)
5. [Behavioral Questions](#behavioral-questions)
6. [Mock Interview Guide](#mock-interview-guide)

---

## Pre-Interview Checklist

### Technical Preparation
- [ ] Review Python fundamentals
- [ ] Practice coding problems
- [ ] Understand data structures (time/space complexity)
- [ ] Know how to optimize solutions
- [ ] Be able to explain your code clearly
- [ ] Practice writing clean code

### On the Day
- [ ] Get good sleep night before
- [ ] Arrive/call 5 minutes early
- [ ] Have quiet, well-lit space
- [ ] Water and pen handy
- [ ] Comfortable clothes
- [ ] Positive mindset

### After Interview
- [ ] Send thank you email
- [ ] Mention specific discussion points
- [ ] Restate your interest
- [ ] Ask next steps

---

## Core Concepts Questions

### Data Types and Structures

**Q1: What are the main data types in Python?**
```python
# Answer: int, float, str, bool, None
# Plus collections: list, tuple, dict, set
x = 5           # int
y = 3.14        # float
z = "hello"     # str
flag = True     # bool
nothing = None  # None
```

**Q2: What's the difference between list and tuple?**
```python
# List - mutable, changeable
my_list = [1, 2, 3]
my_list[0] = 10  # Works

# Tuple - immutable, fixed
my_tuple = (1, 2, 3)
my_tuple[0] = 10  # Error!

# Use list when you need to change
# Use tuple when data should be fixed
```

**Q3: When would you use a set vs a list?**
```python
# Set - unordered, unique items, fast lookup
users_set = {"alice", "bob", "charlie"}
"alice" in users_set  # O(1) - fast

# List - ordered, allows duplicates
users_list = ["alice", "bob", "alice"]
"alice" in users_list  # O(n) - slower

# Use set for: membership testing, removing duplicates
# Use list for: ordered data, allowing duplicates
```

**Q4: Explain list comprehension with an example.**
```python
# Regular way
squares = []
for i in range(5):
    squares.append(i ** 2)

# List comprehension (cleaner)
squares = [i ** 2 for i in range(5)]

# With condition
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]

# Nested
matrix = [[i + j for j in range(3)] for i in range(3)]
```

---

### Functions and Scope

**Q5: What's the difference between *args and **kwargs?**
```python
def my_function(*args, **kwargs):
    # *args - tuple of positional arguments
    print(args)      # (1, 2, 3)
    
    # **kwargs - dictionary of keyword arguments
    print(kwargs)    # {'name': 'Alice', 'age': 25}

my_function(1, 2, 3, name="Alice", age=25)
```

**Q6: What is a closure?**
```python
def outer(x):
    def inner(y):
        return x + y  # Inner function has access to x
    return inner

add_5 = outer(5)
print(add_5(3))  # 8 - 5 is "closed over"

# Useful for: callbacks, partial functions, decorators
```

**Q7: Explain decorators.**
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function")
        result = func(*args, **kwargs)
        print("After function")
        return result
    return wrapper

@my_decorator
def hello():
    print("Hello!")

# Equivalent to: hello = my_decorator(hello)

# Use cases: logging, timing, authentication, caching
```

---

### Object-Oriented Programming

**Q8: What is polymorphism?**
```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

# Polymorphism - same method, different behavior
animals = [Dog(), Cat()]
for animal in animals:
    print(animal.speak())  # Different output per type

# Benefits: flexible, extensible code
```

**Q9: What is inheritance?**
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

# Inheritance - share code, avoid duplication
dog = Dog("Rex")
print(dog.speak())  # "Rex says Woof!"
```

**Q10: Explain the super() function.**
```python
class Parent:
    def method(self):
        return "Parent"

class Child(Parent):
    def method(self):
        parent_result = super().method()
        return f"{parent_result} + Child"

c = Child()
print(c.method())  # "Parent + Child"

# super() - call parent class method
# Use to extend, not replace parent functionality
```

---

## Coding Interview Patterns

### Pattern 1: Sliding Window

**Use When:** Finding substrings, subarrays, or patterns in a sequence

```python
def max_sum_subarray(nums, k):
    """Find max sum of k consecutive elements."""
    if k > len(nums):
        return None
    
    # Initial window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Time: O(n), Space: O(1)
```

### Pattern 2: Two Pointers

**Use When:** Finding pairs, merging arrays, or linked lists

```python
def two_sum(nums, target):
    """Find two numbers that sum to target."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

# Time: O(n), Space: O(1)
```

### Pattern 3: Fast and Slow Pointers

**Use When:** Detecting cycles, finding middle, or sorting linked lists

```python
def has_cycle(head):
    """Detect cycle in linked list."""
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

# Time: O(n), Space: O(1)
```

### Pattern 4: Merge Intervals

**Use When:** Overlapping intervals, meetings, or time slots

```python
def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:
            # Overlapping - merge
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap - add new
            merged.append(current)
    
    return merged

# Time: O(n log n), Space: O(1)
```

### Pattern 5: Hash Map

**Use When:** Frequency, cache, group, or mapping problems

```python
def group_anagrams(words):
    """Group words that are anagrams."""
    anagram_map = {}
    
    for word in words:
        # Sorted word is key
        key = ''.join(sorted(word))
        if key not in anagram_map:
            anagram_map[key] = []
        anagram_map[key].append(word)
    
    return list(anagram_map.values())

# Time: O(n * k log k), Space: O(n * k)
```

---

## Common Interview Problems

### Level 1: Easy

1. **Reverse a string**
2. **Check if palindrome**
3. **Find duplicate in array**
4. **Two sum problem**
5. **Valid parentheses**

### Level 2: Medium

1. **Longest substring without repeating**
2. **Merge k sorted lists**
3. **Binary search tree validation**
4. **Coin change**
5. **Word ladder**

### Level 3: Hard

1. **Median of two sorted arrays**
2. **Regular expression matching**
3. **Trapping rain water**
4. **N-queens problem**
5. **Serialize/deserialize binary tree**

---

## Behavioral Questions

### Question Categories

#### Tell Me About Yourself
Focus on:
- Relevant experience
- Why you're interested in programming
- Key projects or achievements
- What you're looking to learn

```
"I've been learning Python for [timeframe]. 
I'm particularly interested in [specific area] 
because [reason]. My strongest projects are 
[specific examples]. I'm looking for a role 
where I can [growth opportunity]."
```

#### Why This Company?
Research:
- Company's products/services
- Tech stack they use
- Recent news/developments
- Why you align with values

#### Biggest Mistake
Show:
- Self-awareness
- What you learned
- How you improved
- Humility

#### Most Difficult Problem
Demonstrate:
- Problem-solving approach
- Technical knowledge
- Communication
- Perseverance

---

## Mock Interview Guide

### Practice Structure (60 minutes)

1. **Introduction** (5 min)
   - Greeting, small talk
   - Establish rapport

2. **Technical Problem** (40 min)
   - Clarify requirements (5 min)
   - Discuss approach (5 min)
   - Code solution (20 min)
   - Test and optimize (10 min)

3. **Behavioral Question** (10 min)
   - Answer thoughtfully
   - Use specific examples
   - STAR method (Situation, Task, Action, Result)

4. **Your Questions** (5 min)
   - Ask about team/role/company
   - Show genuine interest

### During Interview

**Good Practices:**
- ✓ Think out loud
- ✓ Clarify before coding
- ✓ Test your code
- ✓ Discuss tradeoffs
- ✓ Be honest about edge cases

**Avoid:**
- ✗ Jump into coding immediately
- ✗ Write without testing
- ✗ Ignore edge cases
- ✗ Stay silent while thinking
- ✗ Be defensive about feedback

---

## Interview Do's and Don'ts

### Do's
- ✓ Listen carefully to the problem
- ✓ Ask clarifying questions
- ✓ Start with a clear approach
- ✓ Code step by step
- ✓ Test with examples
- ✓ Explain your thinking
- ✓ Acknowledge mistakes
- ✓ Optimize after getting solution
- ✓ Be enthusiastic
- ✓ Be honest about knowledge

### Don'ts
- ✗ Start coding immediately
- ✗ Write messy code
- ✗ Assume you know the problem
- ✗ Ignore edge cases
- ✗ Stay silent for long periods
- ✗ Memorize solutions
- ✗ Get defensive
- ✗ Pretend to know something you don't
- ✗ Copy code without understanding
- ✗ Forget to test

---

## Time Management in Interviews

### 45-Minute Technical Interview

| Time | Activity | Notes |
|------|----------|-------|
| 0-5 | Understand | Ask clarifying questions |
| 5-10 | Discuss approach | Walk through algorithm |
| 10-35 | Implement | Write and test code |
| 35-40 | Optimize | Improve time/space |
| 40-45 | Discuss | Q&A, next steps |

### Key Timing Tips
- Take 2-3 minutes to fully understand
- Spend 25 minutes coding
- Always leave time to test
- Have a quick optimization strategy

---

## Resources for Interview Prep

### Websites
- LeetCode - leetcode.com (most popular)
- HackerRank - hackerrank.com
- Blind - teamblind.com (interview experiences)
- Glassdoor - glassdoor.com (company reviews)

### Books
- "Cracking the Coding Interview" - Gayle Laakmann McDowell
- "System Design Interview" - Alex Xu

### Videos
- Tech With Tim - Python fundamentals
- Neetcode - Problem explanations
- TechLead - Interview tips

### Practice Platforms
- InterviewBit - Structured learning
- CodeSignal - Practice and assessment
- Pramp - Live mock interviews

---

## Final Tips

### Before Interview
1. **Review fundamentals** - Don't over-memorize
2. **Practice problems** - Do 20-30 coding problems
3. **Understand concepts** - Know WHY, not just WHAT
4. **Research company** - Know their products/tech
5. **Prepare questions** - Have 3-4 genuine questions

### During Interview
1. **Stay calm** - It's okay to think
2. **Communicate** - Explain your process
3. **Be confident** - You're prepared
4. **Stay focused** - Don't jump to wrong conclusions
5. **Ask for help** - If truly stuck, ask for hint

### After Interview
1. **Review performance** - What went well/poorly?
2. **Follow up** - Thank you email
3. **Keep practicing** - Apply lessons learned
4. **Stay positive** - One interview doesn't define you

---

## Common Mistakes to Avoid

1. **Jumping into code too fast**
   - Take time to understand
   - Discuss approach first

2. **Not testing examples**
   - Test happy path
   - Test edge cases
   - Test error cases

3. **Writing unreadable code**
   - Use clear variable names
   - Add comments if needed
   - Proper indentation

4. **Ignoring complexity analysis**
   - Discuss time complexity
   - Discuss space complexity
   - Explain tradeoffs

5. **Not asking questions**
   - Clarify problem statement
   - Ask about constraints
   - Ask about follow-ups

---

## The Interview Philosophy

Remember:
- **Interviews are conversations** - Not interrogations
- **They want you to succeed** - They're hiring
- **It's mutual evaluation** - You're evaluating them too
- **Practice makes perfect** - Each interview helps
- **One bad interview isn't failure** - Learn and move on

---

## Success Indicators

You're well-prepared if you can:
- ✓ Solve medium problems in 30 minutes
- ✓ Discuss time/space complexity confidently
- ✓ Write clean, readable code
- ✓ Handle edge cases naturally
- ✓ Explain your thought process clearly
- ✓ Optimize solutions beyond brute force
- ✓ Answer behavioral questions with examples
- ✓ Ask intelligent questions about the role

---

**Good luck with your interviews! Remember: you've got this! 💪**

---

Last Updated: 2024
Based on actual interview experiences
