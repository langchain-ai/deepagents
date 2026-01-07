"""
MODULE 6 SOLUTIONS
==================

Solutions for Module 6 exercises.
"""

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import csv
import random
import tempfile

# Exercise 1: datetime Module
# ============================

print("EXERCISE 1: datetime Module")
print("-" * 40)

from datetime import datetime, timedelta

# Get current date and time
now = datetime.now()
print(f"Current date/time: {now}")

# Calculate days until next year
current_year = now.year
new_year = datetime(current_year + 1, 1, 1)
days_until_new_year = (new_year - now).days
print(f"Days until next year: {days_until_new_year}")

# Format date as "Month Day, Year"
formatted_date = now.strftime("%B %d, %Y")
print(f"Formatted: {formatted_date}")

print()


# Exercise 2: random Module
# =========================

print("EXERCISE 2: random Module")
print("-" * 40)

def number_guessing_game():
    """Simple number guessing game."""
    secret_number = random.randint(1, 100)
    guesses = 0
    
    print("I'm thinking of a number between 1 and 100")
    print("(Demo mode: secret number is", secret_number, ")")
    
    # Demo: just one guess
    guess = 50
    guesses += 1
    
    if guess < secret_number:
        print(f"Too low! Guess higher.")
    elif guess > secret_number:
        print(f"Too high! Guess lower.")
    else:
        print(f"Correct! You got it in {guesses} guesses!")
    
    return guesses

guesses = number_guessing_game()

print()


# Exercise 3: JSON Operations
# ============================

print("EXERCISE 3: JSON Operations")
print("-" * 40)

# Create Python dictionary
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York",
    "hobbies": ["reading", "coding", "hiking"]
}

# Convert to JSON string
json_string = json.dumps(person, indent=2)
print("Converted to JSON:")
print(json_string)

# Convert back to Python object
parsed = json.loads(json_string)
print(f"\nParsed back - Name: {parsed['name']}, Age: {parsed['age']}")

print()


# Exercise 4: Regular Expressions
# ================================

print("EXERCISE 4: Regular Expressions")
print("-" * 40)

text = """
Contact us at: alice@example.com or bob@company.org
For support: support@help.io
Admin: admin@site.com
"""

# Find all emails
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
emails = re.findall(email_pattern, text)

print("Emails found:")
for email in emails:
    print(f"  - {email}")

print()


# Exercise 5: Collections Counter
# ================================

print("EXERCISE 5: Collections Counter")
print("-" * 40)

text = "the quick brown fox jumps over the lazy dog the fox is quick"
words = text.split()

word_counts = Counter(words)
print(f"Most common 3 words: {word_counts.most_common(3)}")

print()


# Exercise 6: CSV Operations
# ===========================

print("EXERCISE 6: CSV Operations")
print("-" * 40)

# Create CSV data
csv_file = "/tmp/students.csv"

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age", "Grade"])
    writer.writerow(["Alice", "20", "A"])
    writer.writerow(["Bob", "19", "B"])
    writer.writerow(["Charlie", "21", "A"])

# Read CSV data
print("CSV data read:")
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['Name']}: Age {row['Age']}, Grade {row['Grade']}")

print()


# Exercise 7: Path Operations
# ============================

print("EXERCISE 7: Path Operations")
print("-" * 40)

# Create Path object
file_path = Path("example.txt")

print(f"Path: {file_path}")
print(f"Name: {file_path.name}")
print(f"Stem: {file_path.stem}")
print(f"Suffix: {file_path.suffix}")

# Create actual file
file_path.write_text("Hello, World!")
print(f"Content: {file_path.read_text()}")

# Clean up
file_path.unlink()

print()


# Exercise 8: Multiple Modules
# =============================

print("EXERCISE 8: Multiple Modules")
print("-" * 40)

# Combine multiple modules
from datetime import datetime
import json

# Create data with timestamp
data = {
    "event": "User logged in",
    "user": "alice",
    "timestamp": datetime.now().isoformat()
}

# Convert to JSON
log_entry = json.dumps(data, indent=2)
print("Log entry:")
print(log_entry)

print()


# Exercise 9: Create Custom Module
# =================================

print("EXERCISE 9: Create Custom Module")
print("-" * 40)

# Define inline module (normally in separate file)
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract two numbers."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Use the module
print(f"add(5, 3) = {add(5, 3)}")
print(f"subtract(10, 4) = {subtract(10, 4)}")
print(f"multiply(6, 7) = {multiply(6, 7)}")
print(f"divide(20, 4) = {divide(20, 4)}")

print()


# Exercise 10: Virtual Environment
# =================================

print("EXERCISE 10: Virtual Environment")
print("-" * 40)

print("""
Commands to create and use virtual environment:

1. Create virtual environment:
   python -m venv venv

2. Activate (Mac/Linux):
   source venv/bin/activate

3. Activate (Windows):
   venv\\Scripts\\activate

4. Install package:
   pip install requests

5. Save requirements:
   pip freeze > requirements.txt

6. Install from requirements:
   pip install -r requirements.txt

Benefits:
- Isolate dependencies per project
- No conflicts between projects
- Easy to share requirements
- Production-ready setup
""")

print()


# CHALLENGE SOLUTIONS
# ===================

print("=" * 40)
print("CHALLENGE SOLUTIONS")
print("=" * 40 + "\n")

# Challenge 1: API Data Fetching
# ==============================

print("CHALLENGE 1: API Data Fetching")
print("-" * 40)

print("""
Example using requests (if installed):

import requests
import json

response = requests.get('https://jsonplaceholder.typicode.com/posts/1')

if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))
else:
    print(f"Error: {response.status_code}")
""")

print()


# Challenge 2: File System Operations
# ===================================

print("CHALLENGE 2: File System Operations")
print("-" * 40)

def find_python_files(directory):
    """Find all Python files in directory."""
    path = Path(directory)
    python_files = list(path.rglob("*.py"))
    return python_files

# Demo with current directory
python_files = find_python_files(".")
print(f"Found {len(python_files)} Python files")
for f in python_files[:3]:
    print(f"  - {f}")

print()


# Challenge 3: Data Processing
# =============================

print("CHALLENGE 3: Data Processing")
print("-" * 40)

# Create sample CSV
csv_data = "/tmp/data.csv"
with open(csv_data, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Score"])
    writer.writerow(["Alice", "85"])
    writer.writerow(["Bob", "92"])

# Process the data
processed_data = []
with open(csv_data, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        processed_row = {
            "name": row["Name"],
            "score": int(row["Score"]),
            "timestamp": datetime.now().isoformat()
        }
        processed_data.append(processed_row)

# Save as JSON
json_file = "/tmp/processed_data.json"
with open(json_file, 'w') as f:
    json.dump(processed_data, f, indent=2)

print("Processed data saved:")
with open(json_file, 'r') as f:
    print(f.read())

print()
print("=" * 40)
print("You've completed all Module 6 exercises!")
print("=" * 40)
