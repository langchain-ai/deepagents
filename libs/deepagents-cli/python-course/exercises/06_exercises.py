"""
MODULE 6 EXERCISES
==================

Practice working with libraries and modules.
"""

# Exercise 1: datetime Module
# ============================
# Use datetime to:
# - Get current date and time
# - Calculate days until next year
# - Format date as "Month Day, Year"

print("EXERCISE 1: datetime Module")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 2: random Module
# =========================
# Create a simple number guessing game using random

print("EXERCISE 2: random Module")
print("-" * 40)

# YOUR CODE HERE
# def number_guessing_game():

print()


# Exercise 3: JSON Operations
# ============================
# Convert Python dict to JSON and back

print("EXERCISE 3: JSON Operations")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 4: Regular Expressions
# ================================
# Find all email addresses in text

print("EXERCISE 4: Regular Expressions")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 5: Collections Counter
# ================================
# Use Counter to find most common words in text

print("EXERCISE 5: Collections Counter")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 6: CSV Operations
# ===========================
# Read and write CSV data

print("EXERCISE 6: CSV Operations")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 7: Path Operations
# ============================
# Use pathlib to work with file paths

print("EXERCISE 7: Path Operations")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 8: Multiple Modules
# =============================
# Combine multiple modules in one program

print("EXERCISE 8: Multiple Modules")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 9: Create Custom Module
# =================================
# Create a simple module that can be imported

print("EXERCISE 9: Create Custom Module")
print("-" * 40)

print("""
Create a file called my_math.py with:
- add(a, b)
- subtract(a, b)
- multiply(a, b)
- divide(a, b)

Then import and use it in another file.
""")

print()


# Exercise 10: Virtual Environment
# =================================
# Set up project with virtual environment

print("EXERCISE 10: Virtual Environment")
print("-" * 40)

print("""
Commands to run:
1. python -m venv venv
2. source venv/bin/activate (Mac/Linux) or venv\\Scripts\\activate (Windows)
3. pip install requests
4. pip freeze > requirements.txt
5. cat requirements.txt

This creates isolated environment for your project.
""")

print()


# CHALLENGE EXERCISES
# ===================

print("\n" + "=" * 40)
print("CHALLENGE EXERCISES")
print("=" * 40 + "\n")

# Challenge 1: API Data Fetching
# ==============================
# Use requests library to fetch data from API

print("CHALLENGE 1: API Data Fetching")
print("-" * 40)

print("""
If you have requests installed:
pip install requests

Example:
import requests
response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
data = response.json()
print(data)
""")

print()


# Challenge 2: File System Operations
# ===================================
# Walk through directory and find all Python files

print("CHALLENGE 2: File System Operations")
print("-" * 40)

# YOUR CODE HERE

print()


# Challenge 3: Data Processing
# =============================
# Combine json, csv, and datetime for real task

print("CHALLENGE 3: Data Processing")
print("-" * 40)

print("""
Create a program that:
1. Reads data from CSV
2. Converts to JSON
3. Adds timestamp to each record
4. Saves processed JSON
""")

print()

print("=" * 40)
print("Great job! Check solutions/06_solutions.py")
print("=" * 40)
