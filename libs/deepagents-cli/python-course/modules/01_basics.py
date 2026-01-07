"""
MODULE 1: INTRODUCTION TO PYTHON BASICS
========================================

Welcome to Python! This module covers the fundamentals every programmer needs.

Topics:
1. What is Python?
2. Variables and Data Types
3. Basic Operations
4. Strings and String Operations
5. Input and Output
"""

# ============================================================================
# 1. WHAT IS PYTHON?
# ============================================================================
# Python is a simple, readable, powerful programming language.
# It's great for beginners because the syntax is clean and intuitive.

# Everything in Python is an object that has a type and a value.


# ============================================================================
# 2. VARIABLES AND DATA TYPES
# ============================================================================

# Variables are containers for storing values
# Python automatically determines the type for you

# Integer - whole numbers
age = 25
print(f"Age: {age}")
print(f"Type: {type(age)}")

# Float - decimal numbers
height = 5.9
print(f"Height: {height}")
print(f"Type: {type(height)}")

# String - text
name = "Alice"
print(f"Name: {name}")
print(f"Type: {type(name)}")

# Boolean - True or False
is_student = True
print(f"Is Student: {is_student}")
print(f"Type: {type(is_student)}")

# None - the absence of a value
nothing = None
print(f"Nothing: {nothing}")
print(f"Type: {type(nothing)}")


# ============================================================================
# 3. BASIC OPERATIONS
# ============================================================================

# Arithmetic operations
a = 10
b = 3

print("\nArithmetic Operations:")
print(f"{a} + {b} = {a + b}")      # Addition
print(f"{a} - {b} = {a - b}")      # Subtraction
print(f"{a} * {b} = {a * b}")      # Multiplication
print(f"{a} / {b} = {a / b}")      # Division (returns float)
print(f"{a} // {b} = {a // b}")    # Floor division (returns integer)
print(f"{a} % {b} = {a % b}")      # Modulo (remainder)
print(f"{a} ** {b} = {a ** b}")    # Exponentiation

# Comparison operations
print("\nComparison Operations:")
print(f"{a} > {b}: {a > b}")       # Greater than
print(f"{a} < {b}: {a < b}")       # Less than
print(f"{a} >= {b}: {a >= b}")     # Greater than or equal
print(f"{a} <= {b}: {a <= b}")     # Less than or equal
print(f"{a} == {b}: {a == b}")     # Equal to
print(f"{a} != {b}: {a != b}")     # Not equal to

# Logical operations
x = True
y = False

print("\nLogical Operations:")
print(f"{x} and {y}: {x and y}")   # Both must be True
print(f"{x} or {y}: {x or y}")     # At least one must be True
print(f"not {x}: {not x}")         # Inverts the boolean


# ============================================================================
# 4. STRINGS AND STRING OPERATIONS
# ============================================================================

# Strings are sequences of characters
greeting = "Hello, World!"
print(f"\n{greeting}")

# Accessing individual characters (indexing)
# Important: Python uses 0-based indexing!
print(f"First character: {greeting[0]}")      # 'H'
print(f"Last character: {greeting[-1]}")      # '!'
print(f"Character at index 7: {greeting[7]}")  # 'W'

# Slicing strings (getting parts of strings)
print(f"First 5 characters: {greeting[0:5]}")     # 'Hello'
print(f"From index 7 onward: {greeting[7:]}")     # 'World!'
print(f"Every other character: {greeting[::2]}")  # 'Hlowrd'

# String properties
print(f"Length: {len(greeting)}")              # How many characters

# String methods
print(f"Uppercase: {greeting.upper()}")        # Convert to uppercase
print(f"Lowercase: {greeting.lower()}")        # Convert to lowercase
print(f"Contains 'World': {'World' in greeting}")  # Check if substring exists

# String concatenation (combining strings)
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(f"Full name: {full_name}")

# F-strings (formatted strings) - modern way
age = 30
message = f"{first_name} is {age} years old"
print(message)

# String multiplication (repeat strings)
print("=" * 40)  # Print 40 equals signs


# ============================================================================
# 5. INPUT AND OUTPUT
# ============================================================================

# Output with print()
print("\nInput/Output Examples:")
print("This is a simple message")
print("You can", "print", "multiple", "values")

# Input with input()
# Note: Uncomment the lines below to test interactive input
# user_name = input("What is your name? ")
# print(f"Hello, {user_name}!")

# Converting input (input always returns strings)
# user_age = input("How old are you? ")
# user_age = int(user_age)  # Convert string to integer
# print(f"Next year you'll be {user_age + 1}")


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "=" * 40)
print("KEY CONCEPTS SUMMARY")
print("=" * 40)

print("""
1. Variables store values with automatic type detection
2. Main data types: int, float, str, bool, None
3. Operations: arithmetic, comparison, logical
4. Strings are sequences - use indexing and slicing
5. Use print() to output and input() for user input
6. F-strings provide clean formatting
7. Python is 0-indexed (first item is at position 0)
""")

print("=" * 40)
print("Ready for Module 1 Exercises!")
print("=" * 40)
