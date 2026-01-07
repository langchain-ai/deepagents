# Python Debugging Guide & Common Pitfalls

Learn to identify, understand, and fix common Python errors and problems.

## 🔍 Debugging Fundamentals

### Understanding Error Messages

Python errors have three key parts:

```python
Traceback (most recent call last):
  File "program.py", line 3, in <module>
    result = numbers[10]
IndexError: list index out of range
```

1. **Traceback** - Shows where the error occurred (file, line, code)
2. **Error Type** - The kind of error (IndexError, ValueError, etc.)
3. **Message** - Explanation of what went wrong

### Reading Tracebacks

```
Traceback (most recent call last):           # Error occurred
  File "main.py", line 15, in my_function    # In my_function at line 15
    result = calculate(x)                    # This line caused it
  File "main.py", line 8, in calculate       # Called from calculate at line 8
    return 100 / x                           # This did the bad thing
ZeroDivisionError: division by zero          # The actual error
```

Read from **bottom to top** to understand the chain of calls.

---

## 🐛 Common Errors and Solutions

### 1. NameError - Undefined Variable

```python
print(name)  # Error: 'name' is not defined

# Problems:
# - Variable not created
# - Typo in variable name
# - Variable created after use
# - Variable in different scope
```

**Solutions:**
```python
# ✓ Define before use
name = "Alice"
print(name)

# ✓ Check spelling
my_variable = 42
print(my_variable)  # Not: print(my_vvariable)

# ✓ Define at module level if needed globally
x = 10
def my_function():
    print(x)  # This works
```

### 2. TypeError - Wrong Type

```python
result = "5" + 5  # Error: can't concatenate str and int

# Problems:
# - Operations on wrong types
# - Functions called with wrong types
# - No automatic type conversion
```

**Solutions:**
```python
# ✓ Convert to same type
result = "5" + str(5)           # "55"
result = int("5") + 5           # 10

# ✓ Check function requirements
number = int(user_input)        # Convert before use

# ✓ Use isinstance() to check
if isinstance(x, int):
    print(f"x is {x}")
```

### 3. IndexError - Out of Range

```python
items = [1, 2, 3]
print(items[5])  # Error: list index out of range

# Problems:
# - Index too large
# - Index too small (negative)
# - Empty list
# - Off-by-one error
```

**Solutions:**
```python
items = [1, 2, 3]

# ✓ Check length first
if len(items) > 5:
    print(items[5])

# ✓ Use try/except
try:
    print(items[5])
except IndexError:
    print("Index not found")

# ✓ Use safe methods
if items:                   # Check not empty
    print(items[0])

# ✓ Remember 0-based indexing
print(items[0])  # First item (1)
print(items[2])  # Last item in this list (3)
```

### 4. KeyError - Dictionary Key Missing

```python
person = {"name": "Alice", "age": 25}
print(person["email"])  # Error: 'email'

# Problems:
# - Key doesn't exist
# - Typo in key name
# - Key is wrong type
```

**Solutions:**
```python
person = {"name": "Alice", "age": 25}

# ✓ Check if key exists
if "email" in person:
    print(person["email"])

# ✓ Use get() with default
email = person.get("email", "unknown@example.com")

# ✓ Use try/except
try:
    print(person["email"])
except KeyError:
    print("Email not found")

# ✓ Check spelling
print(person["name"])  # Not "Name"
```

### 5. ValueError - Wrong Value

```python
age = int("twenty")  # Error: invalid literal for int()

# Problems:
# - Can't convert value to type
# - Wrong format
# - Out of expected range
```

**Solutions:**
```python
# ✓ Validate before conversion
user_input = "twenty"
try:
    age = int(user_input)
except ValueError:
    print("Please enter a number")
    age = 0

# ✓ Check format first
if user_input.isdigit():
    age = int(user_input)

# ✓ Provide valid options
valid_choices = ["yes", "no"]
choice = input("Yes or no? ")
if choice not in valid_choices:
    print("Invalid choice")
```

### 6. AttributeError - No Attribute

```python
text = "hello"
print(text.upper_case())  # Error: 'str' object has no attribute 'upper_case'

# Problems:
# - Method/attribute doesn't exist
# - Typo in method name
# - Wrong object type
# - None value
```

**Solutions:**
```python
# ✓ Check method name
text = "hello"
print(text.upper())  # Not: upper_case()

# ✓ Verify object type
if hasattr(obj, 'method_name'):
    obj.method_name()

# ✓ Handle None values
if obj is not None:
    print(obj.some_attribute)

# ✓ Use dir() to see available methods
text = "hello"
print(dir(text))  # Shows all methods
```

### 7. ZeroDivisionError - Division by Zero

```python
result = 10 / 0  # Error: division by zero

# Problems:
# - Dividing by zero
# - Dividing by variable that is zero
# - Modulo by zero
```

**Solutions:**
```python
# ✓ Check for zero first
if divisor != 0:
    result = 10 / divisor

# ✓ Use try/except
try:
    result = 10 / divisor
except ZeroDivisionError:
    print("Cannot divide by zero")
    result = 0

# ✓ Provide default
divisor = user_input or 1  # Use 1 if empty
result = 10 / divisor
```

### 8. FileNotFoundError - File Missing

```python
file = open("missing.txt")  # Error: [Errno 2] No such file or directory

# Problems:
# - File doesn't exist
# - Wrong path
# - File was deleted
# - Permissions issue
```

**Solutions:**
```python
import os

# ✓ Check file exists
if os.path.exists("file.txt"):
    with open("file.txt") as f:
        data = f.read()

# ✓ Use try/except
try:
    with open("file.txt") as f:
        data = f.read()
except FileNotFoundError:
    print("File not found")
    data = ""

# ✓ Check path
import os
current_dir = os.getcwd()
full_path = os.path.join(current_dir, "file.txt")
```

### 9. IndentationError - Wrong Indentation

```python
def my_function():
print("Hello")  # Error: expected an indented block

# Problems:
# - Wrong number of spaces
# - Mixed tabs and spaces
# - Missing colon
```

**Solutions:**
```python
# ✓ Indent with consistent spaces (4 spaces recommended)
def my_function():
    print("Hello")      # 4 spaces
    if True:
        print("Inside") # 8 spaces

# ✓ Use IDE that handles indentation
# Most editors auto-indent after colons

# ✓ Never mix tabs and spaces
# Python 3 will error
# Use 4 spaces, not tabs
```

### 10. RecursionError - Infinite Recursion

```python
def infinite():
    infinite()  # Calls itself endlessly

infinite()  # Error: maximum recursion depth exceeded

# Problems:
# - Function calls itself without stopping
# - No base case
# - Base case never reached
```

**Solutions:**
```python
# ✓ Add base case
def countdown(n):
    if n == 0:          # Base case
        return
    print(n)
    countdown(n - 1)    # Recursive case

countdown(5)

# ✓ Make sure recursion ends
def factorial(n):
    if n <= 1:          # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

# ✓ Test with small values first
print(factorial(5))  # Make sure it finishes
```

---

## 🛠️ Debugging Techniques

### 1. Print Debugging

```python
def calculate(x, y):
    print(f"DEBUG: x={x}, y={y}")           # See inputs
    result = x + y
    print(f"DEBUG: result={result}")        # See intermediate
    return result

result = calculate(5, 3)
print(f"DEBUG: final result={result}")      # See output
```

### 2. Using Type Checking

```python
def process(data):
    print(f"Type: {type(data)}")
    print(f"Is list? {isinstance(data, list)}")
    print(f"Is string? {isinstance(data, str)}")
```

### 3. Assertion Testing

```python
def divide(a, b):
    assert b != 0, "Divisor cannot be zero"
    return a / b

divide(10, 0)  # Error: AssertionError: Divisor cannot be zero
```

### 4. Using the Debugger

```python
import pdb

def my_function(x):
    pdb.set_trace()  # Execution stops here
    result = x * 2
    return result

# Debugger commands:
# l - List code
# n - Next line
# p variable - Print variable
# c - Continue
# q - Quit
```

### 5. Logging Instead of Print

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Mutable Default Arguments

```python
# ❌ WRONG - Shares same list across calls
def add_item(item, items=[]):
    items.append(item)
    return items

list1 = add_item(1)     # [1]
list2 = add_item(2)     # [1, 2] - UNEXPECTED!

# ✓ CORRECT - New list each time
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Pitfall 2: Off-by-One Errors

```python
# ❌ WRONG - One too many
for i in range(1, 11):
    print(i)
# Prints: 1 to 10 (forgot range is exclusive at end)

# ✓ CORRECT
for i in range(1, 11):  # Already goes to 10
    print(i)

# ✓ Another way
for i in range(10):      # 0 to 9
    print(i + 1)         # 1 to 10
```

### Pitfall 3: Modifying While Iterating

```python
# ❌ WRONG - Changes list while looping
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # Skips items!

# ✓ CORRECT - Use list comprehension
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]

# ✓ Or create copy
numbers = [1, 2, 3, 4, 5]
for num in numbers.copy():
    if num % 2 == 0:
        numbers.remove(num)
```

### Pitfall 4: Comparing with ==

```python
# ❌ WRONG - Checks value equality, not identity
list1 = [1, 2, 3]
list2 = [1, 2, 3]
if list1 == list2:          # True
    print("Same list")      # Prints, but they're different objects

# Use == for values
# Use 'is' for identity
if list1 is list2:          # False - different objects
    print("Same object")    # Won't print

# Common mistake with None
if x == None:               # Works but not Pythonic
    pass

if x is None:               # Better
    pass
```

### Pitfall 5: Late Binding in Closures

```python
# ❌ WRONG - All functions use last value of x
functions = []
for x in range(5):
    functions.append(lambda: x)

result = [f() for f in functions]
print(result)  # [4, 4, 4, 4, 4] - All are 4!

# ✓ CORRECT - Capture current value
functions = []
for x in range(5):
    functions.append(lambda x=x: x)  # Use default argument

result = [f() for f in functions]
print(result)  # [0, 1, 2, 3, 4]
```

### Pitfall 6: Global vs Local Variables

```python
x = 10  # Global

def my_function():
    x = 5   # Local - doesn't change global
    print(x)

my_function()  # Prints 5
print(x)       # Still 10

# To modify global
x = 10

def my_function():
    global x    # Declare intent
    x = 5       # Now modifies global

my_function()
print(x)  # Now 5
```

### Pitfall 7: Empty List/String Truthiness

```python
# ❌ WRONG
items = []
if items == []:          # Works but awkward
    print("Empty")

# ✓ CORRECT
if not items:            # Cleaner
    print("Empty")

# Also works with strings
text = ""
if not text:             # True for empty string
    print("No text")
```

### Pitfall 8: Forgetting Return Statement

```python
# ❌ WRONG
def calculate(x, y):
    result = x + y
    # Forgot return!

answer = calculate(3, 5)
print(answer)  # None - not 8!

# ✓ CORRECT
def calculate(x, y):
    result = x + y
    return result
```

### Pitfall 9: Using Wrong Scope

```python
# ❌ WRONG
def outer():
    x = 10
    
    def inner():
        print(x)  # Can read x
        x = 5     # But can't modify global x
    
    inner()
    print(x)  # Still 10

# ✓ CORRECT - If you need to modify
def outer():
    x = 10
    
    def inner():
        nonlocal x  # Declare you'll modify outer's x
        x = 5
    
    inner()
    print(x)  # Now 5
```

### Pitfall 10: Type Checking Mistakes

```python
# ❌ WRONG - Fragile
def process(data):
    if type(data) == list:  # Exact type match only
        print("It's a list")

# ✓ CORRECT - More flexible
def process(data):
    if isinstance(data, list):  # Works with subclasses
        print("It's a list-like")
```

---

## 🧪 Testing Your Fixes

### Add Assertions

```python
def divide(a, b):
    assert isinstance(a, (int, float)), "a must be a number"
    assert isinstance(b, (int, float)), "b must be a number"
    assert b != 0, "b cannot be zero"
    return a / b

divide(10, 2)    # Works
divide(10, "2")  # AssertionError
```

### Create Test Cases

```python
def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3
    assert divide(5, 2) == 2.5

test_divide()  # Passes silently
```

### Use Try/Except in Development

```python
try:
    problematic_function()
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
```

---

## ✅ Debugging Checklist

- [ ] Read the error message carefully
- [ ] Find the line number mentioned
- [ ] Add print statements around the error
- [ ] Check variable types with type()
- [ ] Verify variable values
- [ ] Test with simple examples first
- [ ] Check for typos
- [ ] Verify indentation
- [ ] Look for off-by-one errors
- [ ] Check list/dict boundaries
- [ ] Test with edge cases
- [ ] Use try/except for dangerous operations
- [ ] Add assertions for assumptions
- [ ] Use a debugger for complex issues

---

## 📚 Resources

- **Python Docs:** https://docs.python.org/3/
- **Common Errors:** Check module docstrings
- **Stack Overflow:** Search your error message
- **IDLE Debugger:** Built into Python
- **VSCode Debugger:** Use Python extension

---

Remember: **Debugging is a skill that improves with practice.** Every error teaches you something!

Last Updated: 2024
