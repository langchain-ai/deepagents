"""
MODULE 4: FILE I/O AND EXCEPTION HANDLING
==========================================

Learn how to work with files and handle errors gracefully.

Topics:
1. Reading Files
2. Writing Files
3. Working with File Paths
4. Understanding Exceptions
5. Exception Handling with Try/Except
6. Custom Error Handling
"""

# ============================================================================
# 1. READING FILES
# ============================================================================

print("=" * 60)
print("1. READING FILES")
print("=" * 60)

# Basic file reading
# Note: In real use, make sure the file exists first

print("\nExample: Reading a file line by line")
print("Code:")
print("""
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
""")

# Reading specific lines
print("\nExample: Reading lines into a list")
print("""
with open('example.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())
""")

# Reading line by line
print("\nExample: Iterating through lines")
print("""
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())
""")

# Different modes
print("\nFile Opening Modes:")
modes = {
    "'r'": "Read (default) - file must exist",
    "'w'": "Write - creates/overwrites file",
    "'a'": "Append - adds to end of file",
    "'r+'": "Read and write",
    "'x'": "Create - fails if file exists"
}
for mode, description in modes.items():
    print(f"  {mode}: {description}")


# ============================================================================
# 2. WRITING FILES
# ============================================================================

print("\n" + "=" * 60)
print("2. WRITING FILES")
print("=" * 60)

# Create a sample file for demonstration
sample_file_path = "/tmp/sample_course.txt"

print(f"\nWriting to file: {sample_file_path}")

# Writing to a file
with open(sample_file_path, 'w') as file:
    file.write("Python Programming Course\n")
    file.write("Module 4: File I/O and Exceptions\n")
    file.write("=" * 40 + "\n")

print("File created!")

# Appending to a file
print("\nAppending to the file:")
with open(sample_file_path, 'a') as file:
    file.write("\nTopics Covered:\n")
    file.write("1. Reading Files\n")
    file.write("2. Writing Files\n")
    file.write("3. Exception Handling\n")

# Reading what we wrote
print("\nFile contents:")
with open(sample_file_path, 'r') as file:
    content = file.read()
    print(content)

# Using writelines
print("\nWriting multiple lines at once:")
data = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open(sample_file_path, 'w') as file:
    file.writelines(data)

with open(sample_file_path, 'r') as file:
    print(file.read())


# ============================================================================
# 3. WORKING WITH PATHS
# ============================================================================

print("\n" + "=" * 60)
print("3. WORKING WITH FILE PATHS")
print("=" * 60)

import os
from pathlib import Path

# Using os module
print("\nUsing os module:")
print(f"Current directory: {os.getcwd()}")
print(f"Separator: '{os.sep}'")

# Joining paths
path1 = os.path.join("data", "files", "document.txt")
print(f"Joined path: {path1}")

# Path information
print(f"Basename: {os.path.basename(path1)}")
print(f"Directory: {os.path.dirname(path1)}")
print(f"Absolute path: {os.path.abspath(path1)}")

# Using pathlib (modern approach)
print("\nUsing pathlib (modern):")
p = Path(sample_file_path)
print(f"Path object: {p}")
print(f"Name: {p.name}")
print(f"Suffix: {p.suffix}")
print(f"Parent: {p.parent}")
print(f"Exists: {p.exists()}")

# Working with directories
print("\nDirectory operations:")
temp_dir = "/tmp/course_demo"
Path(temp_dir).mkdir(exist_ok=True)
print(f"Created directory: {temp_dir}")

# List directory contents
print(f"Contents of /tmp/: {os.listdir('/tmp')[:5]}...")  # First 5 items


# ============================================================================
# 4. UNDERSTANDING EXCEPTIONS
# ============================================================================

print("\n" + "=" * 60)
print("4. UNDERSTANDING EXCEPTIONS")
print("=" * 60)

print("""
Exceptions are errors that occur during program execution.
They break normal program flow but can be caught and handled.

Common Exceptions:
  ZeroDivisionError - Division by zero
  ValueError - Wrong type of value
  KeyError - Dict key doesn't exist
  IndexError - List index out of range
  FileNotFoundError - File doesn't exist
  TypeError - Wrong type in operation
  AttributeError - Object doesn't have attribute
  NameError - Variable not defined
""")

# Demonstrating exceptions (without crashing)
print("\nExceptions are raised when things go wrong:")
print("Example: 10 / 0 would cause ZeroDivisionError")
print("Example: int('hello') would cause ValueError")
print("Example: d['missing'] would cause KeyError")


# ============================================================================
# 5. EXCEPTION HANDLING WITH TRY/EXCEPT
# ============================================================================

print("\n" + "=" * 60)
print("5. EXCEPTION HANDLING WITH TRY/EXCEPT")
print("=" * 60)

# Basic try/except
print("\nBasic try/except:")
try:
    number = int(input_value := "42")
    result = 100 / number
    print(f"100 / {number} = {result}")
except ValueError:
    print("Error: Could not convert to integer")

# Catching specific exceptions
print("\nCatching multiple exception types:")
try:
    # Simulate different errors
    my_list = [1, 2, 3]
    value = my_list[0]
    print(f"First item: {value}")
except IndexError:
    print("Error: Index out of range")
except ValueError:
    print("Error: Invalid value")

# Using else clause
print("\nUsing try/except/else:")
try:
    number = 10
    result = 100 / number
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
else:
    print(f"Success! 100 / {number} = {result}")

# Using finally clause (always executes)
print("\nUsing try/except/finally:")
try:
    file = open(sample_file_path, 'r')
    content = file.read()
    print(f"Read {len(content)} characters")
except FileNotFoundError:
    print("Error: File not found")
finally:
    print("Cleanup: Closing file if open")
    if 'file' in locals() and not file.closed:
        file.close()

# Catching all exceptions
print("\nCatching any exception with bare except:")
try:
    # Some operation that might fail
    x = 10
except:
    print("Something went wrong!")

# Better: catch Exception class
print("\nCatching Exception (more specific):")
try:
    result = 100 / 0
except Exception as e:
    print(f"An error occurred: {type(e).__name__}: {e}")


# ============================================================================
# 6. CUSTOM ERROR HANDLING
# ============================================================================

print("\n" + "=" * 60)
print("6. CUSTOM ERROR HANDLING")
print("=" * 60)

# Raising exceptions
print("\nRaising custom exceptions:")

def validate_age(age):
    """Validate that age is reasonable."""
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return age

# Using the function
try:
    validate_age(-5)
except ValueError as e:
    print(f"Validation error: {e}")

try:
    validate_age(25)
    print("Age is valid!")
except ValueError as e:
    print(f"Validation error: {e}")

# Creating custom exception classes
print("\nCreating custom exceptions:")

class InsufficientBalanceError(Exception):
    """Raised when account balance is too low."""
    pass

class InvalidPasswordError(Exception):
    """Raised when password is invalid."""
    pass

def withdraw(balance, amount):
    """Withdraw money from account."""
    if amount > balance:
        raise InsufficientBalanceError(
            f"Insufficient balance. Have ${balance}, need ${amount}"
        )
    return balance - amount

# Using custom exceptions
try:
    new_balance = withdraw(100, 150)
except InsufficientBalanceError as e:
    print(f"Transaction failed: {e}")

try:
    new_balance = withdraw(100, 50)
    print(f"Withdrawal successful. New balance: ${new_balance}")
except InsufficientBalanceError as e:
    print(f"Transaction failed: {e}")


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 60)
print("PRACTICAL EXAMPLES")
print("=" * 60)

# Example 1: Safe file reading
print("\nExample 1: Safe file reading with error handling")

def read_file_safe(filename):
    """Read file with error handling."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None

content = read_file_safe(sample_file_path)
if content:
    print(f"File contents: {content[:50]}...")

# Example 2: Type conversion with error handling
print("\nExample 2: Safe type conversion")

def get_integer(prompt):
    """Get integer input from user with retry."""
    while True:
        try:
            value = int(input(f"{prompt} (or 'quit' to exit): "))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

# Example 3: Dictionary access with defaults
print("\nExample 3: Safe dictionary access")

def get_config_value(config, key, default=None):
    """Get config value with fallback."""
    try:
        return config[key]
    except KeyError:
        print(f"Warning: Key '{key}' not found. Using default.")
        return default

config = {"host": "localhost", "port": 8000}
host = get_config_value(config, "host")
timeout = get_config_value(config, "timeout", 30)

print(f"Host: {host}")
print(f"Timeout: {timeout} (default used)")


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS SUMMARY")
print("=" * 60)

summary = """
FILE MODES:
  'r' - Read (default)
  'w' - Write (creates/overwrites)
  'a' - Append (adds to end)

WITH STATEMENT:
  with open(file) as f:
      # Automatically closes file
  File is closed even if error occurs

COMMON METHODS:
  read() - Read entire file as string
  readline() - Read one line
  readlines() - Read all lines as list
  write() - Write string to file
  writelines() - Write list of strings

EXCEPTION HANDLING:
  try: risky code
  except ErrorType: handle specific error
  except: catch any error
  else: runs if no error
  finally: always runs

RAISING EXCEPTIONS:
  raise ValueError("error message")

CUSTOM EXCEPTIONS:
  class MyError(Exception): pass
  raise MyError("message")
"""

print(summary)

print("=" * 60)
print("Ready for Module 4 Exercises!")
print("=" * 60)
