# Python Quick Reference Guide

Quick lookup for syntax and common patterns. Use this while coding!

## 📋 Table of Contents
1. [Variables and Types](#variables-and-types)
2. [Operators](#operators)
3. [Control Flow](#control-flow)
4. [Loops](#loops)
5. [Functions](#functions)
6. [Data Structures](#data-structures)
7. [File I/O](#file-io)
8. [Exceptions](#exceptions)
9. [Classes and OOP](#classes-and-oop)
10. [Common Methods](#common-methods)

---

## Variables and Types

### Creating Variables
```python
# Assignment
name = "Alice"
age = 25
height = 5.9
is_student = True
nothing = None

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0

# Type checking
type(variable)              # Returns type
isinstance(x, int)          # Check if specific type
```

### Type Conversion
```python
int("42")                   # String to integer
float("3.14")              # String to float
str(42)                    # Integer to string
bool(1)                    # Any type to boolean
list("abc")                # String to list: ['a', 'b', 'c']
```

---

## Operators

### Arithmetic
```python
+   # Addition
-   # Subtraction
*   # Multiplication
/   # Division (returns float)
//  # Floor division (returns int)
%   # Modulo (remainder)
**  # Exponentiation
```

### Comparison
```python
==  # Equal to
!=  # Not equal to
>   # Greater than
<   # Less than
>=  # Greater or equal
<=  # Less or equal
```

### Logical
```python
and  # Both must be True
or   # At least one must be True
not  # Inverts boolean
```

### Assignment
```python
=   # Assign
+=  # Add and assign: x += 5 (x = x + 5)
-=  # Subtract and assign
*=  # Multiply and assign
/=  # Divide and assign
```

### Membership
```python
in      # Check if in collection
not in  # Check if not in collection
```

---

## Control Flow

### If Statement
```python
if condition:
    # Code runs if True

if condition:
    # Code if True
else:
    # Code if False

if condition1:
    # First check
elif condition2:
    # Second check
else:
    # None matched
```

### Ternary Operator
```python
value = x if condition else y
```

---

## Loops

### For Loop
```python
# Loop through sequence
for item in sequence:
    print(item)

# Loop with index
for i in range(5):          # 0 to 4
    print(i)

for i in range(1, 6):       # 1 to 5
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)

# With enumerate (get index)
for index, item in enumerate(list):
    print(f"{index}: {item}")

# Loop control
break       # Exit loop
continue    # Skip to next iteration
else:       # Runs if loop completes normally
```

### While Loop
```python
while condition:
    # Runs while condition is True
    if something:
        break       # Exit loop
    if other:
        continue    # Skip to next iteration
```

### List Comprehension
```python
[x for x in range(5)]                    # [0, 1, 2, 3, 4]
[x**2 for x in range(5)]                 # [0, 1, 4, 9, 16]
[x for x in range(10) if x % 2 == 0]    # [0, 2, 4, 6, 8]
{x: x**2 for x in range(3)}              # {0: 0, 1: 1, 2: 4}
{x for x in range(5)}                    # {0, 1, 2, 3, 4}
```

---

## Functions

### Definition and Calling
```python
def function_name():
    return None

def add(a, b):
    return a + b

result = add(3, 5)

# Default parameters
def greet(name="Friend"):
    return f"Hello, {name}!"

# *args - variable number of positional arguments
def sum_all(*args):
    return sum(args)

sum_all(1, 2, 3, 4)  # Works with any number

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Lambda functions (anonymous)
square = lambda x: x**2
square(5)           # Returns 25
```

### Function Documentation
```python
def my_function(param):
    """This is a docstring."""
    pass

help(my_function)   # Show docstring
```

---

## Data Structures

### Lists
```python
# Create
list1 = [1, 2, 3, 4, 5]
list2 = ["a", "b", "c"]
list3 = [1, "hello", 3.14, True]
empty = []

# Access
list1[0]            # First item: 1
list1[-1]           # Last item: 5
list1[1:3]          # Slice: [2, 3]
list1[::2]          # Every other: [1, 3, 5]

# Modify
list1[0] = 10       # Change item
list1.append(6)     # Add to end
list1.insert(0, 0)  # Insert at position
list1.extend([7,8]) # Add multiple
list1.remove(2)     # Remove by value
list1.pop(0)        # Remove by index
list1.clear()       # Remove all

# Methods
len(list1)          # Length
list1.count(2)      # Count occurrences
list1.index(3)      # Find index
list1.sort()        # Sort in place
list1.reverse()     # Reverse in place
sorted(list1)       # Return sorted copy
```

### Tuples
```python
# Create (immutable)
tuple1 = (1, 2, 3)
tuple2 = (1,)       # Single element needs comma
empty = ()

# Access (same as lists)
tuple1[0]           # 1
tuple1[1:3]         # (2, 3)

# Unpacking
x, y, z = tuple1
a, *rest = (1, 2, 3, 4)  # a=1, rest=[2,3,4]
```

### Dictionaries
```python
# Create
dict1 = {"name": "Alice", "age": 25}
dict2 = dict(name="Bob", age=30)
empty = {}

# Access
dict1["name"]           # "Alice"
dict1.get("age")        # 25
dict1.get("city", "NY") # Default if not found

# Modify
dict1["email"] = "alice@example.com"  # Add/update
del dict1["age"]        # Delete
dict1.pop("name")       # Remove and return

# Iterate
for key in dict1:       # Keys
for value in dict1.values():  # Values
for key, value in dict1.items():  # Both

# Methods
dict1.keys()            # All keys
dict1.values()          # All values
dict1.items()           # All key-value pairs
dict1.clear()           # Remove all
```

### Sets
```python
# Create (unique items)
set1 = {1, 2, 3, 4}
set2 = set([3, 4, 5, 6])
empty = set()

# Add/Remove
set1.add(5)             # Add item
set1.remove(2)          # Remove (error if not found)
set1.discard(2)         # Remove (no error if not found)

# Operations
set1 | set2             # Union
set1 & set2             # Intersection
set1 - set2             # Difference
set1 ^ set2             # Symmetric difference
```

### Strings
```python
# Create
str1 = "Hello"
str2 = 'World'
str3 = """Multi
line
string"""

# Access
str1[0]                 # 'H'
str1[1:4]              # 'ell'
str1[-1]               # 'o'

# Methods
str1.upper()           # "HELLO"
str1.lower()           # "hello"
str1.capitalize()      # "Hello"
str1.replace("H", "J") # "Jello"
str1.split(",")        # Split by delimiter
",".join(["a", "b"])   # "a,b"
str1.strip()           # Remove whitespace
str1.startswith("He")  # True
str1.endswith("o")     # True
str1.find("l")         # 2 (first index)
str1.count("l")        # 2 (count occurrences)

# Formatting
f"Hello {name}"        # F-string (preferred)
"Hello {}".format(name)  # Format method
"Hello %s" % name      # Old style (avoid)
```

---

## File I/O

### Reading Files
```python
# Read entire file
with open("file.txt", "r") as f:
    content = f.read()

# Read line by line
with open("file.txt", "r") as f:
    for line in f:
        print(line.strip())

# Read all lines as list
with open("file.txt", "r") as f:
    lines = f.readlines()
```

### Writing Files
```python
# Write (overwrites file)
with open("file.txt", "w") as f:
    f.write("Hello\n")
    f.write("World\n")

# Append (adds to end)
with open("file.txt", "a") as f:
    f.write("More text\n")

# Write multiple lines
with open("file.txt", "w") as f:
    f.writelines(["Line 1\n", "Line 2\n"])
```

### File Paths
```python
import os
from pathlib import Path

# OS module
os.getcwd()              # Current directory
os.path.join("a", "b")   # Join paths
os.path.exists(path)     # Check if exists
os.listdir(".")          # List directory

# Pathlib (modern)
p = Path("file.txt")
p.exists()               # Check exists
p.parent                 # Parent directory
p.name                   # Filename
p.stem                   # Filename without extension
p.suffix                 # File extension
```

---

## Exceptions

### Raising Exceptions
```python
raise ValueError("Invalid value")
raise TypeError("Wrong type")
raise Exception("Generic error")
```

### Catching Exceptions
```python
# Single exception
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Multiple exceptions
try:
    x = int("abc")
except (ValueError, TypeError):
    print("Error with value or type")

# Catch any exception
try:
    risky_code()
except Exception as e:
    print(f"Error: {e}")

# With else (if no error)
try:
    x = 10 / 2
except ZeroDivisionError:
    print("Error")
else:
    print(f"Result: {x}")

# With finally (always runs)
try:
    file = open("file.txt")
finally:
    file.close()
```

### Custom Exceptions
```python
class MyError(Exception):
    pass

try:
    raise MyError("Custom message")
except MyError as e:
    print(f"Caught: {e}")
```

---

## Classes and OOP

### Class Definition
```python
class Dog:
    """Dog class."""
    
    species = "Canis familiaris"  # Class attribute
    
    def __init__(self, name, age):
        """Constructor."""
        self.name = name            # Instance attribute
        self.age = age
    
    def bark(self):
        """Method."""
        return f"{self.name} says: Woof!"
    
    def __str__(self):
        """String representation."""
        return f"{self.name} ({self.age} years)"

# Create instance
dog = Dog("Buddy", 3)
print(dog.bark())
```

### Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Sound"

class Cat(Animal):
    def speak(self):  # Override
        return "Meow"

cat = Cat("Whiskers")
print(cat.speak())  # "Meow"
```

### Properties
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
```

### Special Methods
```python
class MyClass:
    def __init__(self):          # Constructor
        pass
    
    def __str__(self):           # String representation
        return "MyClass instance"
    
    def __repr__(self):          # Developer representation
        return "MyClass()"
    
    def __len__(self):           # len(obj)
        return 0
    
    def __getitem__(self, index):  # obj[index]
        return None
    
    def __setitem__(self, index, value):  # obj[index] = value
        pass
    
    def __eq__(self, other):     # obj == other
        return True
    
    def __lt__(self, other):     # obj < other
        return False
```

---

## Common Methods

### String Methods
```python
.upper()         # Uppercase
.lower()         # Lowercase
.capitalize()    # First letter capital
.title()         # Title case
.strip()         # Remove whitespace
.replace(old, new)  # Replace text
.split(sep)      # Split into list
.join(list)      # Join list into string
.find(substr)    # Find index
.startswith(s)   # Check start
.endswith(s)     # Check end
```

### List Methods
```python
.append(item)    # Add to end
.insert(i, item) # Insert at index
.extend(list)    # Add multiple
.remove(item)    # Remove by value
.pop(index)      # Remove by index
.clear()         # Remove all
.sort()          # Sort in place
.reverse()       # Reverse in place
.count(item)     # Count occurrences
.index(item)     # Find index
```

### Dictionary Methods
```python
.keys()          # Get keys
.values()        # Get values
.items()         # Get key-value pairs
.get(key, default)  # Safe access
.pop(key)        # Remove and return
.clear()         # Remove all
.update(dict)    # Merge dictionaries
```

### Set Methods
```python
.add(item)       # Add item
.remove(item)    # Remove (error if missing)
.discard(item)   # Remove (no error)
.clear()         # Remove all
.union(set)      # Union (|)
.intersection(set)  # Intersection (&)
.difference(set) # Difference (-)
```

---

## Built-in Functions

```python
len(obj)         # Length
min(iterable)    # Minimum
max(iterable)    # Maximum
sum(iterable)    # Sum
sorted(iterable) # Sorted list
reversed(iterable)  # Reversed iterator
enumerate(iterable) # Index and value
zip(iter1, iter2)   # Combine iterables
map(func, iterable) # Apply function
filter(func, iterable)  # Filter items
all(iterable)    # All True?
any(iterable)    # Any True?
abs(x)          # Absolute value
round(x, digits) # Round
pow(x, y)       # Power
isinstance(obj, type)  # Type check
callable(obj)   # Is callable?
```

---

## Common Imports

```python
# Math
import math
math.sqrt(9)
math.pi
math.ceil(4.3)
math.floor(4.7)

# Random
import random
random.randint(1, 10)
random.choice([1, 2, 3])
random.shuffle(list)

# Date and Time
from datetime import datetime
now = datetime.now()
now.year
now.month

# Operating System
import os
os.getcwd()
os.path.exists("file")

# JSON
import json
json.dumps(dict)        # Dict to string
json.loads(string)      # String to dict

# Regular Expressions
import re
re.match(pattern, string)
re.findall(pattern, string)
re.sub(pattern, repl, string)
```

---

## Code Patterns

### Safe Dictionary Access
```python
# Get with default
value = dict.get(key, default_value)

# Check before access
if key in dict:
    value = dict[key]
```

### Safe List Access
```python
# Check bounds
if 0 <= index < len(list):
    item = list[index]

# Use try/except
try:
    item = list[index]
except IndexError:
    item = None
```

### Safe Type Conversion
```python
try:
    number = int(user_input)
except ValueError:
    number = 0
```

### File Handling
```python
# Always use with statement
with open("file.txt") as f:
    data = f.read()
# File automatically closes
```

### Class with Validation
```python
class Account:
    def __init__(self, balance):
        if balance < 0:
            raise ValueError("Balance cannot be negative")
        self.balance = balance
```

---

## Tips and Best Practices

✅ **Do:**
- Use meaningful variable names
- Add docstrings to functions
- Use with statements for files
- Handle exceptions appropriately
- Use list/dict comprehensions
- Follow PEP 8 style guide

❌ **Don't:**
- Use bare except clauses
- Catch all exceptions generically
- Use mutable default arguments
- Ignore exceptions silently
- Use global variables excessively
- Write functions longer than ~50 lines

---

## Quick Debugging

```python
# Print values
print(f"x = {x}, y = {y}")

# Check type
print(type(variable))

# Check if None
if variable is None:
    print("It's None")

# Check if in collection
if item in list:
    print("Found it")

# Show exception details
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

---

**Remember:** This is a quick reference. For detailed explanations, see the course modules!

Last Updated: 2024  
Python Version: 3.8+
