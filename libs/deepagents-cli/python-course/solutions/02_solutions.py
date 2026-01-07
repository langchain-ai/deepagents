"""
MODULE 2 SOLUTIONS
==================

Solutions for Module 2 exercises.
"""

# Exercise 1: Simple If/Else
# ==========================

print("EXERCISE 1: Simple If/Else")
print("-" * 40)

def check_number_sign(number):
    if number > 0:
        return "Positive"
    elif number < 0:
        return "Negative"
    else:
        return "Zero"

print(f"10 is {check_number_sign(10)}")
print(f"-5 is {check_number_sign(-5)}")
print(f"0 is {check_number_sign(0)}")
print()


# Exercise 2: Age Category
# ========================

print("EXERCISE 2: Age Category")
print("-" * 40)

def get_age_category(age):
    if age < 13:
        return "Child"
    elif age < 20:
        return "Teen"
    elif age < 65:
        return "Adult"
    else:
        return "Senior"

print(f"Age 8: {get_age_category(8)}")
print(f"Age 15: {get_age_category(15)}")
print(f"Age 30: {get_age_category(30)}")
print(f"Age 70: {get_age_category(70)}")
print()


# Exercise 3: For Loop - Print Numbers
# =====================================

print("EXERCISE 3: For Loop - Print Numbers")
print("-" * 40)

for i in range(1, 11):
    print(i)
print()


# Exercise 4: For Loop - Sum Numbers
# ===================================

print("EXERCISE 4: For Loop - Sum Numbers")
print("-" * 40)

total = 0
for i in range(1, 101):
    total += i

print(f"Sum of 1 to 100: {total}")
print()


# Exercise 5: For Loop - Multiplication Table
# ============================================

print("EXERCISE 5: For Loop - Multiplication Table")
print("-" * 40)

def mult_table(n):
    for i in range(1, 11):
        print(f"{n}*{i}={n*i}")

print("Table of 3:")
mult_table(3)
print()


# Exercise 6: While Loop - Countdown
# ===================================

print("EXERCISE 6: While Loop - Countdown")
print("-" * 40)

count = 5
while count >= 1:
    print(count)
    count -= 1
print()


# Exercise 7: Function - Rectangle Area
# ======================================

print("EXERCISE 7: Function - Rectangle Area")
print("-" * 40)

def rectangle_area(length, width):
    return length * width

area = rectangle_area(5, 3)
print(f"Rectangle 5x3 area: {area}")
print()


# Exercise 8: Function - Maximum of Three Numbers
# ================================================

print("EXERCISE 8: Function - Maximum of Three Numbers")
print("-" * 40)

def max_of_three(a, b, c):
    return max(a, b, c)

result = max_of_three(15, 7, 20)
print(f"Max of 15, 7, 20: {result}")
print()


# Exercise 9: Function - Check Even/Odd
# ======================================

print("EXERCISE 9: Function - Check Even/Odd")
print("-" * 40)

def is_even(number):
    return number % 2 == 0

print(f"8 is even: {is_even(8)}")
print(f"7 is even: {is_even(7)}")
print()


# Exercise 10: Function - String Repeat
# ======================================

print("EXERCISE 10: Function - String Repeat")
print("-" * 40)

def repeat_string(text, n):
    return text * n

result = repeat_string("Ha", 3)
print(f"'Ha' repeated 3 times: {result}")
print()


# CHALLENGE EXERCISES
# ===================

print("=" * 40)
print("CHALLENGE SOLUTIONS")
print("=" * 40 + "\n")

# Challenge 1: Fibonacci Sequence
# ===============================

print("CHALLENGE 1: Fibonacci")
print("-" * 40)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

print(f"Fibonacci(5): {fibonacci(5)}")
print(f"Fibonacci(8): {fibonacci(8)}")

# Print first 10 Fibonacci numbers
print("First 10 Fibonacci numbers:")
for i in range(1, 11):
    print(f"  F({i}) = {fibonacci(i)}")
print()


# Challenge 2: Password Validator
# ===============================

print("CHALLENGE 2: Password Validator")
print("-" * 40)

def is_strong_password(password):
    # Check length
    if len(password) < 8:
        return False
    
    # Check for at least one digit
    has_digit = False
    for char in password:
        if char.isdigit():
            has_digit = True
            break
    
    if not has_digit:
        return False
    
    # Check for at least one uppercase
    has_upper = False
    for char in password:
        if char.isupper():
            has_upper = True
            break
    
    return has_upper

print(f"'abc123' is strong: {is_strong_password('abc123')}")
print(f"'Abc123' is strong: {is_strong_password('Abc123')}")
print(f"'ABC123XY' is strong: {is_strong_password('ABC123XY')}")
print()


# Challenge 3: Print Triangle
# ============================

print("CHALLENGE 3: Print Triangle")
print("-" * 40)

def print_triangle(n):
    for i in range(1, n + 1):
        print("*" * i)

print("Triangle of size 5:")
print_triangle(5)

print()
print("=" * 40)
print("Well done! Move to Module 3")
print("=" * 40)
