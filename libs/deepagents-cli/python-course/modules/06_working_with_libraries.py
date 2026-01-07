"""
MODULE 6: WORKING WITH LIBRARIES AND MODULES
==============================================

Learn to use Python's powerful built-in and third-party libraries.

Topics:
1. What are Modules and Libraries
2. Importing Modules
3. Exploring Built-in Modules
4. Working with Popular Libraries
5. Creating Your Own Modules
6. Virtual Environments and pip
"""

# ============================================================================
# 1. WHAT ARE MODULES AND LIBRARIES?
# ============================================================================

print("=" * 60)
print("1. WHAT ARE MODULES AND LIBRARIES?")
print("=" * 60)

print("""
Module: A file containing Python code (.py file)
  - Can define functions, classes, variables
  - Can be imported and used in other files

Library/Package: A collection of modules
  - Built-in: Come with Python (os, sys, math, etc.)
  - Third-party: Downloaded with pip (requests, pandas, etc.)

Why use libraries?
  - Don't reinvent the wheel
  - Use tested, optimized code
  - Reduce development time
  - Focus on your logic, not implementations
""")


# ============================================================================
# 2. IMPORTING MODULES
# ============================================================================

print("\n" + "=" * 60)
print("2. IMPORTING MODULES")
print("=" * 60)

# Import entire module
import math

print(f"\nUsing math module:")
print(f"  math.pi = {math.pi}")
print(f"  math.sqrt(16) = {math.sqrt(16)}")
print(f"  math.ceil(4.3) = {math.ceil(4.3)}")

# Import specific items
from math import pi, sqrt, cos

print(f"\nImporting specific items:")
print(f"  pi = {pi}")
print(f"  sqrt(25) = {sqrt(25)}")

# Import with alias
import json as j

data = j.dumps({"name": "Alice", "age": 25})
print(f"\nUsing alias:")
print(f"  json.dumps: {data}")

# Import all (generally not recommended)
# from math import *  # Don't do this in production


# ============================================================================
# 3. EXPLORING BUILT-IN MODULES
# ============================================================================

print("\n" + "=" * 60)
print("3. EXPLORING BUILT-IN MODULES")
print("=" * 60)

# datetime module
from datetime import datetime, timedelta

print("\nWorking with dates and times:")
now = datetime.now()
print(f"  Current date/time: {now}")
print(f"  Year: {now.year}")
print(f"  Month: {now.month}")
print(f"  Day: {now.day}")

tomorrow = now + timedelta(days=1)
print(f"  Tomorrow: {tomorrow.date()}")

# random module
import random

print("\nGenerating random values:")
print(f"  Random int (1-10): {random.randint(1, 10)}")
print(f"  Random float (0-1): {random.random():.4f}")
print(f"  Random choice: {random.choice(['apple', 'banana', 'cherry'])}")

items = [1, 2, 3, 4, 5]
random.shuffle(items)
print(f"  Shuffled list: {items}")

# os module (operating system)
import os

print("\nOperating system operations:")
print(f"  Current directory: {os.getcwd()}")
print(f"  Path separator: {os.sep}")
print(f"  Home directory: {os.path.expanduser('~')}")
print(f"  File exists: {os.path.exists('/etc')}")

# pathlib (modern way to work with paths)
from pathlib import Path

print("\nModern path handling with pathlib:")
p = Path("example.txt")
print(f"  Path object: {p}")
print(f"  Name: {p.name}")
print(f"  Suffix: {p.suffix}")
print(f"  Parent: {p.parent}")


# ============================================================================
# 4. WORKING WITH POPULAR LIBRARIES
# ============================================================================

print("\n" + "=" * 60)
print("4. WORKING WITH POPULAR LIBRARIES")
print("=" * 60)

# json - Working with JSON data
print("\nJSON module (data interchange):")
data = {
    "users": [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30}
    ]
}

json_string = json.dumps(data, indent=2)
print("  Converted to JSON:")
print(json_string)

parsed = json.loads(json_string)
print(f"  Parsed back: {parsed['users'][0]['name']}")

# collections module
from collections import Counter, defaultdict

print("\nCollections module (specialized data structures):")

# Counter - count items
words = "hello world hello python hello"
word_count = Counter(words.split())
print(f"  Word frequencies: {word_count}")
print(f"  Most common 2: {word_count.most_common(2)}")

# defaultdict - dict with default values
scores = defaultdict(int)
scores['Alice'] += 10
scores['Bob'] += 5
scores['Alice'] += 5
print(f"  Default dict: {dict(scores)}")

# itertools - Iteration tools
from itertools import combinations, permutations

print("\nItertools module (iteration tools):")
items = [1, 2, 3]
print(f"  Combinations: {list(combinations(items, 2))}")
print(f"  Permutations (first 3): {list(permutations(items, 2))[:3]}")

# re - Regular expressions
import re

print("\nRegular expressions module:")
text = "My email is alice@example.com and bob@test.org"
pattern = r"[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+"
emails = re.findall(pattern, text)
print(f"  Found emails: {emails}")

# csv module
import csv
from io import StringIO

print("\nCSV module:")
csv_data = "Name,Age,City\nAlice,25,NYC\nBob,30,LA"
reader = csv.DictReader(StringIO(csv_data))
rows = list(reader)
print(f"  Parsed CSV: {rows[0]}")


# ============================================================================
# 5. CREATING YOUR OWN MODULES
# ============================================================================

print("\n" + "=" * 60)
print("5. CREATING YOUR OWN MODULES")
print("=" * 60)

print("""
To create a module:
1. Create a .py file with functions/classes
2. Import it in another file

Example: math_utils.py
  def add(a, b):
      return a + b
  
  def multiply(a, b):
      return a * b

Usage: 
  from math_utils import add, multiply
  result = add(3, 5)

Module structure:
  my_project/
  ├── __init__.py          # Makes it a package
  ├── utils.py             # Your module
  ├── main.py              # Uses the module
  └── data.py              # Another module
""")

# Create example module dynamically
example_module_code = '''
"""Example module for demonstration."""

def greet(name):
    """Greet someone."""
    return f"Hello, {name}!"

def calculate_discount(price, discount_percent):
    """Calculate price after discount."""
    return price * (1 - discount_percent / 100)

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self.result
    
    def multiply(self, x):
        self.result *= x
        return self.result
'''

print("Example module code:")
print(example_module_code)


# ============================================================================
# 6. VIRTUAL ENVIRONMENTS AND PIP
# ============================================================================

print("\n" + "=" * 60)
print("6. VIRTUAL ENVIRONMENTS AND PIP")
print("=" * 60)

print("""
pip: Package installer for Python
  - Command: pip install package_name
  - Common packages:
    * requests - HTTP library
    * pandas - Data analysis
    * numpy - Numerical computing
    * flask - Web framework
    * pytest - Testing framework

Virtual Environment: Isolated Python environment
  - Separate dependencies per project
  - No conflicts between projects
  - Good practice for all projects

Creating virtual environment:
  python -m venv venv
  
Activating:
  - Windows: venv\\Scripts\\activate
  - Mac/Linux: source venv/bin/activate

Installing packages:
  pip install requests
  pip install pandas numpy
  
Saving requirements:
  pip freeze > requirements.txt
  
Installing from requirements:
  pip install -r requirements.txt

Best practice:
  Always use virtual environments!
""")


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 60)
print("PRACTICAL EXAMPLES")
print("=" * 60)

# Example 1: Working with multiple modules
print("\nExample 1: Data processing")
from datetime import datetime
import statistics

temperatures = [20, 22, 19, 25, 23, 21, 20]
print(f"  Temperatures: {temperatures}")
print(f"  Average: {statistics.mean(temperatures):.1f}°C")
print(f"  Median: {statistics.median(temperatures):.1f}°C")
print(f"  Std Dev: {statistics.stdev(temperatures):.2f}")
print(f"  Recorded: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Example 2: Configuration management
print("\nExample 2: Configuration as JSON")
config = {
    "app_name": "My App",
    "version": "1.0.0",
    "settings": {
        "debug": True,
        "timeout": 30
    }
}

config_json = json.dumps(config, indent=2)
print("  Config JSON:")
for line in config_json.split('\n'):
    print(f"    {line}")

# Example 3: File operations
print("\nExample 3: File operations")
from pathlib import Path
import tempfile

# Create temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    file_path = Path(tmpdir) / "test.txt"
    file_path.write_text("Hello, World!")
    content = file_path.read_text()
    print(f"  Written and read: {content}")
    print(f"  File exists: {file_path.exists()}")


# ============================================================================
# COMMON MODULES REFERENCE
# ============================================================================

print("\n" + "=" * 60)
print("COMMON MODULES REFERENCE")
print("=" * 60)

modules_info = """
Built-in Modules (no installation needed):

1. math - Mathematical functions
   - sin, cos, sqrt, log, etc.
   - constants like pi, e

2. random - Random number generation
   - randint(), random(), choice(), shuffle()

3. datetime - Date and time
   - datetime, date, time, timedelta

4. os - Operating system operations
   - getcwd(), listdir(), path operations

5. sys - System-specific parameters
   - argv (command line arguments)
   - exit() (exit program)

6. json - JSON data handling
   - dumps() (to string), loads() (from string)

7. re - Regular expressions
   - match(), search(), findall(), sub()

8. collections - Specialized containers
   - Counter, defaultdict, OrderedDict

9. itertools - Iteration tools
   - combinations, permutations, chain

10. pathlib - Object-oriented path handling
    - Path objects instead of strings

11. csv - Reading/writing CSV files
    - DictReader, DictWriter

12. tempfile - Temporary files/directories
    - TemporaryFile, TemporaryDirectory

Popular Third-Party Packages:

1. requests - HTTP library
   pip install requests
   
2. pandas - Data analysis
   pip install pandas
   
3. numpy - Numerical computing
   pip install numpy
   
4. matplotlib - Data visualization
   pip install matplotlib
   
5. flask - Web framework
   pip install flask
   
6. pytest - Testing framework
   pip install pytest
"""

print(modules_info)


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS SUMMARY")
print("=" * 60)

summary = """
Module: Python file with code you can import

Import Statements:
  import math                      # Import entire module
  from math import sqrt            # Import specific function
  from math import sqrt as s       # Import with alias
  import math as m                 # Module alias

Using imported code:
  math.sqrt(16)                    # Use with module prefix
  sqrt(16)                         # Use directly if imported

pip: Package manager
  pip install package_name         # Install
  pip list                         # Show installed
  pip uninstall package_name       # Remove

Virtual Environment:
  python -m venv env_name          # Create
  source env_name/bin/activate     # Activate (Mac/Linux)
  pip install -r requirements.txt  # Install dependencies

Best Practices:
  ✓ Use virtual environments
  ✓ Keep requirements.txt updated
  ✓ Import only what you need
  ✓ Use meaningful import names
  ✓ Organize imports by type
"""

print(summary)

print("=" * 60)
print("Ready for Module 6 Exercises!")
print("=" * 60)
