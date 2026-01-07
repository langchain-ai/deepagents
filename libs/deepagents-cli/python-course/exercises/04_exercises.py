"""
MODULE 4 EXERCISES
==================

Practice file I/O and exception handling.
"""

# Exercise 1: Write to File
# =========================
# Write a list of favorite foods to a file, one per line

print("EXERCISE 1: Write to File")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 2: Read from File
# ===========================
# Read the file from Exercise 1 and print each food with a number

print("EXERCISE 2: Read from File")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 3: Append to File
# ===========================
# Append more foods to the file from Exercise 1

print("EXERCISE 3: Append to File")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 4: Count Lines in File
# ================================
# Create a function that counts the number of lines in a file

print("EXERCISE 4: Count Lines in File")
print("-" * 40)

# YOUR CODE HERE
# def count_lines(filename):

print()


# Exercise 5: Try/Except - ValueError
# ====================================
# Ask user for a number and handle ValueError if input is invalid

print("EXERCISE 5: Try/Except - ValueError")
print("-" * 40)

try:
    user_input = "not_a_number"  # Simulate user input
    number = int(user_input)
    print(f"Number: {number}")
except ValueError:
    # YOUR CODE HERE
    pass

print()


# Exercise 6: Try/Except - FileNotFoundError
# ===========================================
# Try to open a non-existent file and handle the error

print("EXERCISE 6: Try/Except - FileNotFoundError")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 7: Try/Except/Else
# ============================
# Use try/except/else to safely divide two numbers

print("EXERCISE 7: Try/Except/Else")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 8: Try/Except/Finally
# ===============================
# Read a file with try/except/finally (always close file)

print("EXERCISE 8: Try/Except/Finally")
print("-" * 40)

# YOUR CODE HERE

print()


# Exercise 9: Multiple Exception Types
# =====================================
# Handle both ValueError and IndexError in same try block

print("EXERCISE 9: Multiple Exception Types")
print("-" * 40)

try:
    my_list = [1, 2, 3]
    # Try to access element and convert to int
    result = int(my_list[10])  # Will raise IndexError
except IndexError:
    # YOUR CODE HERE
    pass
except ValueError:
    # YOUR CODE HERE
    pass

print()


# Exercise 10: Create Custom Exception
# ======================================
# Define a custom exception and raise it

print("EXERCISE 10: Create Custom Exception")
print("-" * 40)

# YOUR CODE HERE
# class InvalidEmailError(Exception):
#     pass

print()


# CHALLENGE EXERCISES
# ===================

print("\n" + "=" * 40)
print("CHALLENGE EXERCISES")
print("=" * 40 + "\n")

# Challenge 1: Read CSV File
# ==========================
# Create and read a CSV file (comma-separated values)

print("CHALLENGE 1: Read CSV File")
print("-" * 40)

# YOUR CODE HERE

print()


# Challenge 2: Word Counter
# ==========================
# Read a file and count occurrences of each word

print("CHALLENGE 2: Word Counter")
print("-" * 40)

# YOUR CODE HERE

print()


# Challenge 3: Log File Parser
# ============================
# Read a log file and extract error lines (lines containing "ERROR")

print("CHALLENGE 3: Log File Parser")
print("-" * 40)

# YOUR CODE HERE

print()

print("=" * 40)
print("Great job! Check solutions/04_solutions.py")
print("=" * 40)
