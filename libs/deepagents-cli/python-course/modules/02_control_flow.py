"""
MODULE 2: CONTROL FLOW AND FUNCTIONS
=====================================

Learn how to make decisions in your code and create reusable functions.

Topics:
1. If/Elif/Else Statements
2. Comparison and Logical Operators (Review)
3. Loops: For and While
4. Functions: Definition and Calling
5. Function Parameters and Return Values
"""

# ============================================================================
# 1. IF/ELIF/ELSE STATEMENTS
# ============================================================================

print("=" * 60)
print("1. IF/ELIF/ELSE STATEMENTS")
print("=" * 60)

# Basic if statement
age = 18

if age >= 18:
    print(f"You are {age} years old. You are an adult.")

# if/else statement
score = 45

if score >= 60:
    print(f"Score {score}: PASS")
else:
    print(f"Score {score}: FAIL")

# if/elif/else statement (multiple conditions)
temperature = 15

if temperature > 25:
    print(f"Temperature {temperature}°C: It's hot! 🌞")
elif temperature > 15:
    print(f"Temperature {temperature}°C: It's warm.")
elif temperature > 0:
    print(f"Temperature {temperature}°C: It's cold.")
else:
    print(f"Temperature {temperature}°C: It's freezing! 🥶")

# Nested if statements
username = "alice"
password = "secret123"
entered_username = "alice"
entered_password = "secret123"

if entered_username == username:
    if entered_password == password:
        print("Login successful!")
    else:
        print("Invalid password")
else:
    print("Invalid username")


# ============================================================================
# 2. LOGICAL OPERATORS (AND, OR, NOT)
# ============================================================================

print("\n" + "=" * 60)
print("2. LOGICAL OPERATORS")
print("=" * 60)

# AND operator - both conditions must be True
x = 10
y = 20

if x > 5 and y > 15:
    print(f"Both conditions are true: x > 5 AND y > 15")

# OR operator - at least one condition must be True
day = "Saturday"

if day == "Saturday" or day == "Sunday":
    print(f"{day} is a weekend! 🎉")

# NOT operator - inverts the boolean value
is_raining = False

if not is_raining:
    print("It's not raining, let's go outside!")


# ============================================================================
# 3. FOR LOOPS
# ============================================================================

print("\n" + "=" * 60)
print("3. FOR LOOPS")
print("=" * 60)

# Loop through numbers using range()
print("\nCounting from 1 to 5:")
for i in range(1, 6):  # 1 to 5 (6 is excluded)
    print(f"  {i}", end=" ")
print()

# Loop through a string
print("\nLooping through a string:")
word = "Python"
for letter in word:
    print(f"  {letter}", end=" ")
print()

# Using range with step
print("\nCountdown from 10 to 1 (by 2s):")
for i in range(10, 0, -2):
    print(f"  {i}", end=" ")
print()

# Loop with index (enumerate)
print("\nLooping with index:")
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"  {index}: {fruit}")

# Nested loops
print("\nNested loops (multiplication table):")
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i}*{j}={i*j}", end="  ")
    print()


# ============================================================================
# 4. WHILE LOOPS
# ============================================================================

print("\n" + "=" * 60)
print("4. WHILE LOOPS")
print("=" * 60)

# Simple while loop
print("\nCounting with while loop:")
count = 1
while count <= 5:
    print(f"  Count: {count}")
    count += 1  # Same as: count = count + 1

# While loop with break (exit the loop early)
print("\nSearching for a number:")
number = 0
while True:
    number += 1
    print(f"  Checking {number}...", end="")
    if number == 3:
        print(" Found it!")
        break  # Exit the loop

# While loop with continue (skip to next iteration)
print("\nPrinting only even numbers:")
number = 0
while number < 10:
    number += 1
    if number % 2 != 0:  # If odd, skip it
        continue
    print(f"  {number}", end=" ")
print()


# ============================================================================
# 5. FUNCTIONS
# ============================================================================

print("\n" + "=" * 60)
print("5. FUNCTIONS")
print("=" * 60)

# Function definition and calling
def greet():
    """Simple function with no parameters."""
    print("Hello, World!")

greet()  # Calling the function

# Function with parameters
def greet_person(name):
    """Function that takes one parameter."""
    print(f"Hello, {name}!")

greet_person("Alice")
greet_person("Bob")

# Function with multiple parameters
def add(a, b):
    """Function that takes two parameters."""
    result = a + b
    print(f"{a} + {b} = {result}")

add(5, 3)
add(10, 20)

# Function with return value
def multiply(a, b):
    """Function that returns a value."""
    return a * b

product = multiply(4, 5)
print(f"4 * 5 = {product}")

# Function with default parameters
def greet_with_title(name, title="Friend"):
    """Function with default parameter value."""
    print(f"Hello, {title} {name}!")

greet_with_title("Alice")           # Uses default "Friend"
greet_with_title("Bob", "Professor") # Uses custom "Professor"

# Function with multiple return values
def get_coordinates():
    """Function that returns multiple values."""
    x = 10
    y = 20
    return x, y

coord_x, coord_y = get_coordinates()
print(f"Coordinates: ({coord_x}, {coord_y})")


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 60)
print("PRACTICAL EXAMPLES")
print("=" * 60)

# Example 1: Temperature converter
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

temp_c = 25
temp_f = celsius_to_fahrenheit(temp_c)
print(f"\n{temp_c}°C = {temp_f}°F")

# Example 2: Check if number is prime
def is_prime(number):
    """Check if a number is prime."""
    if number < 2:
        return False
    
    for i in range(2, number):
        if number % i == 0:
            return False
    
    return True

print(f"\nIs 17 prime? {is_prime(17)}")
print(f"Is 12 prime? {is_prime(12)}")

# Example 3: Factorial using loop
def factorial(n):
    """Calculate factorial using a loop."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(f"\nFactorial of 5: {factorial(5)}")
print(f"Factorial of 6: {factorial(6)}")

# Example 4: Sum of numbers in a range
def sum_range(start, end):
    """Sum all numbers from start to end (inclusive)."""
    total = 0
    for i in range(start, end + 1):
        total += i
    return total

print(f"\nSum from 1 to 10: {sum_range(1, 10)}")
print(f"Sum from 5 to 15: {sum_range(5, 15)}")


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS SUMMARY")
print("=" * 60)

print("""
1. IF/ELIF/ELSE control program flow based on conditions
2. FOR loops iterate through sequences or ranges
3. WHILE loops repeat as long as a condition is True
4. BREAK exits a loop early, CONTINUE skips to next iteration
5. FUNCTIONS are reusable blocks of code
6. Parameters are inputs to functions
7. Return values are outputs from functions
8. Default parameters provide optional values
9. Functions improve code organization and reusability
""")

print("=" * 60)
print("Ready for Module 2 Exercises!")
print("=" * 60)
