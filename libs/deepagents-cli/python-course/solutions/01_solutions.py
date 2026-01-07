"""
MODULE 1 SOLUTIONS
==================

These are the solutions to Module 1 exercises.
Compare with your code and learn from any differences.
"""

# Exercise 1: Variables and Types
# ================================
print("EXERCISE 1: Variables and Types")
print("-" * 40)

name = "Alice"
age = 25
height = 1.75
like_programming = True

print(f"Name: {name}, Type: {type(name)}")
print(f"Age: {age}, Type: {type(age)}")
print(f"Height: {height}, Type: {type(height)}")
print(f"Like Programming: {like_programming}, Type: {type(like_programming)}")
print()


# Exercise 2: Arithmetic
# =====================
print("EXERCISE 2: Arithmetic")
print("-" * 40)

price = 25.99
tax_rate = 0.08

tax_amount = price * tax_rate
total_price = price + tax_amount

print(f"Original Price: ${price}")
print(f"Tax Amount: ${tax_amount:.2f}")
print(f"Total Price: ${total_price:.2f}")
print()


# Exercise 3: String Operations
# =============================
print("EXERCISE 3: String Operations")
print("-" * 40)

text = "Python Programming"

print(f"Length: {len(text)}")
print(f"First 6 characters: {text[0:6]}")
print(f"Last 11 characters: {text[-11:]}")
print(f"Uppercase: {text.upper()}")
print(f"With addition: {text} is fun")
print()


# Exercise 4: Indexing and Slicing
# ================================
print("EXERCISE 4: Indexing and Slicing")
print("-" * 40)

sentence = "Learning Python is awesome"

print(f"Character at index 0: '{sentence[0]}'")
print(f"Character at index 9: '{sentence[9]}'")
print(f"Characters 0 to 8: '{sentence[0:9]}'")
print(f"Every other character from index 1: '{sentence[1::2]}'")
print(f"Last 7 characters: '{sentence[-7:]}'")
print()


# Exercise 5: Mixed Operations
# ============================
print("EXERCISE 5: Mixed Operations")
print("-" * 40)

score1, score2, score3, score4, score5 = 85, 92, 78, 95, 88

total_score = score1 + score2 + score3 + score4 + score5
average_score = total_score / 5
is_passing = average_score >= 70

print(f"Total Score: {total_score}")
print(f"Average Score: {average_score}")
print(f"Is Passing (>= 70): {is_passing}")
print()


# Exercise 6: String Formatting
# =============================
print("EXERCISE 6: String Formatting")
print("-" * 40)

name = "Bob"
age = 28
city = "New York"

print(f"{name} is {age} years old and lives in {city}")
print()


# Exercise 7: Comparisons
# =======================
print("EXERCISE 7: Comparisons")
print("-" * 40)

num1 = 15
num2 = 10

print(f"num1 > num2: {num1 > num2}")
print(f"num1 == num2: {num1 == num2}")
print(f"num1 != num2: {num1 != num2}")
print(f"(num1 + num2) > 20: {(num1 + num2) > 20}")
print(f"(num1 - num2) < 5: {(num1 - num2) < 5}")
print()


# Exercise 8: Boolean Logic
# =========================
print("EXERCISE 8: Boolean Logic")
print("-" * 40)

is_sunny = True
is_warm = True
have_free_time = False

good_weather = is_sunny and is_warm
can_go_out = good_weather and have_free_time
should_stay_home = (not good_weather) or (not have_free_time)

print(f"Good Weather (sunny AND warm): {good_weather}")
print(f"Can Go Out (good weather AND free time): {can_go_out}")
print(f"Should Stay Home (NOT good or NOT free): {should_stay_home}")
print()


# CHALLENGE EXERCISES (Optional)
# ==============================

print("=" * 40)
print("CHALLENGE SOLUTIONS")
print("=" * 40 + "\n")

# Challenge 1: Password Validator
# ===============================
print("CHALLENGE 1: Password Validator")
print("-" * 40)

password = "MyPass123"
is_long_enough = len(password) >= 8
masked_password = "*" * len(password)

print(f"Password: {password}")
print(f"Length: {len(password)}")
print(f"Is 8+ characters: {is_long_enough}")
print(f"Masked: {masked_password}")
print()


# Challenge 2: BMI Calculator
# ===========================
print("CHALLENGE 2: BMI Calculator")
print("-" * 40)

height = 1.75
weight = 70

bmi = weight / (height ** 2)
is_healthy = 18.5 <= bmi <= 24.9

print(f"Height: {height}m, Weight: {weight}kg")
print(f"BMI: {bmi:.2f}")
print(f"Is Healthy Range (18.5-24.9): {is_healthy}")
print()

print("=" * 40)
print("Well done! Move on to Module 2 when ready")
print("=" * 40)
