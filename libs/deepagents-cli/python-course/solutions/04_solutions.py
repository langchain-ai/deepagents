"""
MODULE 4 SOLUTIONS
==================

Solutions for Module 4 exercises.
"""

import os

# Exercise 1: Write to File
# =========================

print("EXERCISE 1: Write to File")
print("-" * 40)

foods = ["Pizza", "Pasta", "Sushi", "Tacos", "Burger"]
filename = "/tmp/foods.txt"

with open(filename, 'w') as f:
    for food in foods:
        f.write(food + "\n")

print(f"Wrote {len(foods)} foods to {filename}")
print()


# Exercise 2: Read from File
# ===========================

print("EXERCISE 2: Read from File")
print("-" * 40)

with open(filename, 'r') as f:
    lines = f.readlines()
    for i, food in enumerate(lines, 1):
        print(f"{i}. {food.strip()}")

print()


# Exercise 3: Append to File
# ===========================

print("EXERCISE 3: Append to File")
print("-" * 40)

more_foods = ["Ramen", "Steak"]

with open(filename, 'a') as f:
    for food in more_foods:
        f.write(food + "\n")

print(f"Appended {len(more_foods)} more foods")

with open(filename, 'r') as f:
    print(f"Total foods now: {len(f.readlines())}")

print()


# Exercise 4: Count Lines in File
# ================================

print("EXERCISE 4: Count Lines in File")
print("-" * 40)

def count_lines(filename):
    try:
        with open(filename, 'r') as f:
            return len(f.readlines())
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return 0

line_count = count_lines(filename)
print(f"Number of lines: {line_count}")
print()


# Exercise 5: Try/Except - ValueError
# ====================================

print("EXERCISE 5: Try/Except - ValueError")
print("-" * 40)

try:
    user_input = "not_a_number"
    number = int(user_input)
    print(f"Number: {number}")
except ValueError:
    print(f"Error: '{user_input}' is not a valid integer")

print()


# Exercise 6: Try/Except - FileNotFoundError
# ===========================================

print("EXERCISE 6: Try/Except - FileNotFoundError")
print("-" * 40)

try:
    with open("/tmp/nonexistent_file.txt", 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("Error: File not found. Using default content.")

print()


# Exercise 7: Try/Except/Else
# ============================

print("EXERCISE 7: Try/Except/Else")
print("-" * 40)

try:
    num1 = 10
    num2 = 2
    result = num1 / num2
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
else:
    print(f"{num1} / {num2} = {result}")

print()


# Exercise 8: Try/Except/Finally
# ===============================

print("EXERCISE 8: Try/Except/Finally")
print("-" * 40)

try:
    with open(filename, 'r') as f:
        content = f.read()
        print(f"Read {len(content)} characters")
except FileNotFoundError:
    print("Error: File not found")
finally:
    print("File handling complete")

print()


# Exercise 9: Multiple Exception Types
# =====================================

print("EXERCISE 9: Multiple Exception Types")
print("-" * 40)

try:
    my_list = [1, 2, 3]
    result = int(my_list[10])
except IndexError:
    print("Error: List index out of range")
except ValueError:
    print("Error: Could not convert value to integer")

print()


# Exercise 10: Create Custom Exception
# ======================================

print("EXERCISE 10: Create Custom Exception")
print("-" * 40)

class InvalidEmailError(Exception):
    """Raised when email format is invalid."""
    pass

def validate_email(email):
    if "@" not in email or "." not in email:
        raise InvalidEmailError(f"Invalid email: {email}")
    return email

try:
    validate_email("alice@example.com")
    print("Email is valid!")
except InvalidEmailError as e:
    print(f"Error: {e}")

try:
    validate_email("invalid_email")
except InvalidEmailError as e:
    print(f"Error: {e}")

print()


# CHALLENGE EXERCISES
# ===================

print("=" * 40)
print("CHALLENGE SOLUTIONS")
print("=" * 40 + "\n")

# Challenge 1: Read CSV File
# ==========================

print("CHALLENGE 1: Read CSV File")
print("-" * 40)

csv_file = "/tmp/scores.csv"

# Create CSV file
with open(csv_file, 'w') as f:
    f.write("Name,Math,English,Science\n")
    f.write("Alice,85,90,88\n")
    f.write("Bob,92,88,95\n")
    f.write("Charlie,78,92,80\n")

# Read and parse CSV
with open(csv_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    print(f"Headers: {header}")
    
    for line in lines[1:]:
        values = line.strip().split(',')
        name, math, eng, sci = values
        print(f"{name}: Math={math}, English={eng}, Science={sci}")

print()


# Challenge 2: Word Counter
# ==========================

print("CHALLENGE 2: Word Counter")
print("-" * 40)

text_file = "/tmp/sample_text.txt"

# Create sample text file
with open(text_file, 'w') as f:
    f.write("Python is great. Python is fun. Python is powerful.\n")
    f.write("Learning Python helps you code better.\n")

# Count words
with open(text_file, 'r') as f:
    text = f.read().lower()
    words = text.split()
    
    word_count = {}
    for word in words:
        # Remove punctuation
        clean_word = word.strip('.,!?;:')
        word_count[clean_word] = word_count.get(clean_word, 0) + 1

print("Word counts:")
for word, count in sorted(word_count.items(), key=lambda x: x[1], reverse=True):
    if count > 1:
        print(f"  {word}: {count}")

print()


# Challenge 3: Log File Parser
# ============================

print("CHALLENGE 3: Log File Parser")
print("-" * 40)

log_file = "/tmp/app.log"

# Create sample log file
with open(log_file, 'w') as f:
    f.write("[INFO] Application started\n")
    f.write("[DEBUG] Loading config file\n")
    f.write("[ERROR] Config file not found\n")
    f.write("[INFO] Using default config\n")
    f.write("[ERROR] Connection timeout\n")
    f.write("[DEBUG] Retrying connection\n")

# Extract error lines
with open(log_file, 'r') as f:
    lines = f.readlines()

print("Errors found:")
error_count = 0
for i, line in enumerate(lines, 1):
    if "ERROR" in line:
        print(f"  Line {i}: {line.strip()}")
        error_count += 1

print(f"\nTotal errors: {error_count}")

print()
print("=" * 40)
print("Excellent progress! Move to Module 5")
print("=" * 40)
