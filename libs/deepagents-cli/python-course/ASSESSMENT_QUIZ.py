#!/usr/bin/env python3
"""
Python Course Assessment Quiz System

Interactive quiz to test knowledge after completing modules.
Provides immediate feedback and progress tracking.
"""

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Question:
    """Represents a single quiz question."""
    id: int
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: str
    module: int


class Quiz:
    """Python course assessment quiz."""
    
    # Quiz database
    QUESTIONS = [
        # Module 1 Questions
        Question(
            id=1,
            question="What is the output of: print(type(5.0))?",
            options=["<class 'int'>", "<class 'float'>", "<class 'str'>", "<class 'number'>"],
            correct_answer=1,
            explanation="5.0 is a float because it has a decimal point.",
            difficulty="Easy",
            module=1
        ),
        Question(
            id=2,
            question="Which is NOT a valid Python variable name?",
            options=["_my_var", "MyVar", "my-var", "myVar"],
            correct_answer=2,
            explanation="Variable names cannot contain hyphens. Use underscores instead.",
            difficulty="Easy",
            module=1
        ),
        Question(
            id=3,
            question="What does 10 // 3 evaluate to?",
            options=["3", "3.333...", "4", "3.0"],
            correct_answer=0,
            explanation="// is floor division, which returns an integer.",
            difficulty="Easy",
            module=1
        ),
        
        # Module 2 Questions
        Question(
            id=4,
            question="What will this print?\nif False:\n    print('A')\nelse:\n    print('B')",
            options=["A", "B", "AB", "Nothing"],
            correct_answer=1,
            explanation="The condition is False, so the else block executes.",
            difficulty="Easy",
            module=2
        ),
        Question(
            id=5,
            question="How many times does this loop run?\nfor i in range(5):\n    print(i)",
            options=["4", "5", "6", "Error"],
            correct_answer=1,
            explanation="range(5) generates 0, 1, 2, 3, 4 - which is 5 numbers.",
            difficulty="Easy",
            module=2
        ),
        Question(
            id=6,
            question="What is the purpose of 'return' in a function?",
            options=[
                "Exit the program",
                "Send output back to caller",
                "Print to console",
                "Store in memory"
            ],
            correct_answer=1,
            explanation="return sends a value back to the code that called the function.",
            difficulty="Medium",
            module=2
        ),
        
        # Module 3 Questions
        Question(
            id=7,
            question="What is list[::2] if list = [1, 2, 3, 4, 5]?",
            options=["[1, 3]", "[1, 3, 5]", "[2, 4]", "[3]"],
            correct_answer=1,
            explanation="[::2] means start to end, step by 2. So: 1, 3, 5.",
            difficulty="Medium",
            module=3
        ),
        Question(
            id=8,
            question="Which data structure would you use for fast membership testing?",
            options=["List", "Tuple", "Set", "String"],
            correct_answer=2,
            explanation="Sets use hash tables for O(1) membership testing.",
            difficulty="Medium",
            module=3
        ),
        Question(
            id=9,
            question="What's the output of dict[key] if key doesn't exist?",
            options=["None", "KeyError", "Empty dict", "False"],
            correct_answer=1,
            explanation="Accessing missing key raises KeyError. Use .get() to return None.",
            difficulty="Medium",
            module=3
        ),
        
        # Module 4 Questions
        Question(
            id=10,
            question="What exception is raised by int('hello')?",
            options=["TypeError", "ValueError", "NameError", "AttributeError"],
            correct_answer=1,
            explanation="ValueError is raised when converting invalid string to int.",
            difficulty="Medium",
            module=4
        ),
        Question(
            id=11,
            question="When is the 'finally' block executed?",
            options=[
                "Only if exception occurs",
                "Only if no exception",
                "Always, regardless of exception",
                "Never"
            ],
            correct_answer=2,
            explanation="finally always executes, even if exception or return occurs.",
            difficulty="Medium",
            module=4
        ),
        
        # Module 5 Questions
        Question(
            id=12,
            question="What is the purpose of 'self' in a class method?",
            options=[
                "Local variable",
                "Reference to the instance",
                "Reference to the class",
                "Global variable"
            ],
            correct_answer=1,
            explanation="self refers to the specific instance of the class.",
            difficulty="Medium",
            module=5
        ),
        Question(
            id=13,
            question="What is inheritance in OOP?",
            options=[
                "Creating multiple instances",
                "Child class getting parent's properties",
                "Global variable sharing",
                "Making code shorter"
            ],
            correct_answer=1,
            explanation="Inheritance allows child classes to reuse parent class code.",
            difficulty="Medium",
            module=5
        ),
        
        # Advanced Questions
        Question(
            id=14,
            question="What is a list comprehension used for?",
            options=[
                "Creating lists concisely",
                "Looping without lists",
                "Sorting lists",
                "Checking if item in list"
            ],
            correct_answer=0,
            explanation="List comprehensions provide a concise way to create lists.",
            difficulty="Medium",
            module=3
        ),
        Question(
            id=15,
            question="Which statement is best practice?",
            options=[
                "import *",
                "from module import function",
                "import module as m",
                "Both b and c are good"
            ],
            correct_answer=3,
            explanation="Specific imports are clearer. Aliases are acceptable for clarity.",
            difficulty="Hard",
            module=6
        ),
    ]
    
    def __init__(self):
        """Initialize quiz."""
        self.score = 0
        self.total = 0
        self.current_question = 0
        self.results = []
    
    def display_question(self, question: Question) -> None:
        """Display a question and its options."""
        print(f"\n{'='*60}")
        print(f"Question {question.id}: {question.difficulty}")
        print(f"Module {question.module}")
        print(f"{'='*60}")
        print(f"\n{question.question}\n")
        
        for i, option in enumerate(question.options, 1):
            print(f"  {i}. {option}")
    
    def get_user_answer(self, num_options: int) -> Optional[int]:
        """Get answer from user with validation."""
        while True:
            try:
                answer = input(f"\nYour answer (1-{num_options}): ").strip()
                answer_int = int(answer) - 1
                if 0 <= answer_int < num_options:
                    return answer_int
                print(f"Please enter a number between 1 and {num_options}")
            except ValueError:
                print("Please enter a valid number")
    
    def check_answer(self, question: Question, user_answer: int) -> bool:
        """Check if answer is correct and show feedback."""
        is_correct = user_answer == question.correct_answer
        
        print(f"\n{'-'*60}")
        if is_correct:
            print("✓ CORRECT!")
            self.score += 1
        else:
            print("✗ INCORRECT")
            correct = question.options[question.correct_answer]
            print(f"   Correct answer: {correct}")
        
        print(f"\nExplanation: {question.explanation}")
        print(f"{'-'*60}")
        
        self.results.append({
            "question_id": question.id,
            "user_answer": question.options[user_answer],
            "correct_answer": question.options[question.correct_answer],
            "is_correct": is_correct
        })
        
        return is_correct
    
    def run_quiz(self, module: Optional[int] = None) -> None:
        """Run the quiz, optionally filtering by module."""
        questions = self.QUESTIONS
        
        if module:
            questions = [q for q in questions if q.module == module]
        
        self.score = 0
        self.total = len(questions)
        self.results = []
        
        print(f"\n{'='*60}")
        print("PYTHON COURSE ASSESSMENT QUIZ")
        print(f"{'='*60}")
        
        if module:
            print(f"Module {module} Quiz - {self.total} Questions")
        else:
            print(f"Full Course Quiz - {self.total} Questions")
        
        print(f"{'='*60}\n")
        
        for i, question in enumerate(questions, 1):
            self.display_question(question)
            user_answer = self.get_user_answer(len(question.options))
            self.check_answer(question, user_answer)
            
            if i < len(questions):
                input("\nPress Enter for next question...")
        
        self.show_results()
    
    def show_results(self) -> None:
        """Show final results and score."""
        percentage = (self.score / self.total * 100) if self.total > 0 else 0
        
        print(f"\n{'='*60}")
        print("QUIZ RESULTS")
        print(f"{'='*60}")
        print(f"Score: {self.score}/{self.total}")
        print(f"Percentage: {percentage:.1f}%")
        
        if percentage >= 90:
            rating = "Excellent! You've mastered this content."
        elif percentage >= 80:
            rating = "Good! You have solid understanding."
        elif percentage >= 70:
            rating = "Fair. Review the explanations for missed questions."
        else:
            rating = "Keep practicing. Review the modules again."
        
        print(f"Rating: {rating}")
        print(f"{'='*60}\n")
    
    def show_summary(self) -> None:
        """Show detailed results summary."""
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(self.results, 1):
            status = "✓" if result["is_correct"] else "✗"
            print(f"{i}. {status} Question {result['question_id']}")
            if not result["is_correct"]:
                print(f"   Your answer: {result['user_answer']}")
                print(f"   Correct answer: {result['correct_answer']}")
        
        print(f"\n{'='*60}\n")


def main():
    """Main quiz interface."""
    quiz = Quiz()
    
    while True:
        print(f"\n{'='*60}")
        print("PYTHON COURSE ASSESSMENT SYSTEM")
        print(f"{'='*60}")
        print("\nOptions:")
        print("  1. Full course quiz (all modules)")
        print("  2. Module 1 quiz")
        print("  3. Module 2 quiz")
        print("  4. Module 3 quiz")
        print("  5. Module 4 quiz")
        print("  6. Module 5 quiz")
        print("  7. Module 6 quiz")
        print("  8. View previous results")
        print("  9. Exit")
        print(f"{'='*60}")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == "1":
            quiz.run_quiz()
        elif choice in ["2", "3", "4", "5", "6", "7"]:
            module_num = int(choice) - 1
            quiz.run_quiz(module=module_num)
        elif choice == "8":
            if quiz.results:
                quiz.show_summary()
            else:
                print("\nNo previous results to display.")
        elif choice == "9":
            print("\nThank you for using the Python Course Assessment System!")
            break
        else:
            print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     PYTHON COURSE ASSESSMENT QUIZ SYSTEM                   ║
    ║                                                            ║
    ║  Test your knowledge with comprehensive quizzes for each  ║
    ║  module. Get immediate feedback and detailed explanations.║
    ║                                                            ║
    ║  Features:                                                 ║
    ║  - Module-specific quizzes                                 ║
    ║  - Full course assessment                                  ║
    ║  - Instant feedback                                        ║
    ║  - Score tracking                                          ║
    ║  - Progress reporting                                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    main()
