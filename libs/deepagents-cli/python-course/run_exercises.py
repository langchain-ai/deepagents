#!/usr/bin/env python3
"""
Exercise Runner - Test and run all course exercises

This script helps you run exercises and check solutions interactively.

Usage:
    python run_exercises.py              # Show menu
    python run_exercises.py 1            # Run Module 1
    python run_exercises.py 1 check      # Check solutions
"""

import sys
import os
import subprocess
from pathlib import Path


class ExerciseRunner:
    """Manages exercise execution and solution checking."""
    
    def __init__(self):
        """Initialize the runner."""
        self.base_path = Path(__file__).parent
        self.modules = {
            1: "Introduction to Python Basics",
            2: "Control Flow and Functions",
            3: "Data Structures",
            4: "File I/O and Exception Handling",
            5: "Object-Oriented Programming Basics"
        }
    
    def show_menu(self):
        """Display main menu."""
        print("\n" + "=" * 70)
        print("PYTHON COURSE - EXERCISE RUNNER")
        print("=" * 70)
        print("\nAvailable Modules:\n")
        
        for num, title in self.modules.items():
            print(f"  {num}. {title}")
        
        print("\nUsage:")
        print("  python run_exercises.py <module> [action]")
        print("\nActions:")
        print("  (none/exercise) - Run exercises file")
        print("  solution        - Run solutions file")
        print("  compare         - Show diff between exercise and solution")
        print("  both            - Run exercises then solutions")
        print("\nExamples:")
        print("  python run_exercises.py 1              # Run Module 1 exercises")
        print("  python run_exercises.py 2 solution     # Run Module 2 solutions")
        print("  python run_exercises.py 3 compare      # Compare Module 3")
        print("\n" + "=" * 70 + "\n")
    
    def get_file_path(self, module_num, file_type):
        """
        Get path to exercise or solution file.
        
        Args:
            module_num (int): Module number (1-5)
            file_type (str): 'exercise' or 'solution'
            
        Returns:
            Path: File path
        """
        if file_type == "exercise":
            directory = "exercises"
            filename = f"0{module_num}_exercises.py"
        elif file_type == "solution":
            directory = "solutions"
            filename = f"0{module_num}_solutions.py"
        elif file_type == "module":
            directory = "modules"
            filename = f"0{module_num}_*.py"
        
        path = self.base_path / directory / filename
        return path
    
    def run_file(self, file_path):
        """
        Run a Python file.
        
        Args:
            file_path (Path): File to run
            
        Returns:
            bool: Success status
        """
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return False
        
        try:
            print(f"\nRunning: {file_path.name}")
            print("-" * 70)
            subprocess.run([sys.executable, str(file_path)], check=True)
            print("-" * 70)
            return True
        except subprocess.CalledProcessError:
            print(f"Error running {file_path.name}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def compare_files(self, module_num):
        """
        Show comparison between exercise and solution.
        
        Args:
            module_num (int): Module number
        """
        exercise_path = self.get_file_path(module_num, "exercise")
        solution_path = self.get_file_path(module_num, "solution")
        
        if not exercise_path.exists() or not solution_path.exists():
            print("Error: Files not found")
            return
        
        try:
            # Use diff command if available
            subprocess.run(
                ["diff", str(exercise_path), str(solution_path)],
                check=False
            )
        except FileNotFoundError:
            print("Note: diff command not available")
            print("\nTo compare, open both files in your editor:")
            print(f"  Exercise: {exercise_path}")
            print(f"  Solution: {solution_path}")
    
    def list_module_contents(self, module_num):
        """List what's in a module."""
        module_path = self.base_path / "modules" / f"0{module_num}_*.py"
        
        # Find the actual file
        module_files = list(self.base_path.glob(f"modules/0{module_num}_*.py"))
        if not module_files:
            print(f"Module {module_num} not found")
            return
        
        module_file = module_files[0]
        print(f"\nModule: {module_file.name}")
        print(f"Title: {self.modules.get(module_num, 'Unknown')}")
        print(f"Size: {module_file.stat().st_size:,} bytes")
        
        # Show first few lines
        with open(module_file) as f:
            lines = f.readlines()
            print("\nFirst 20 lines:")
            for i, line in enumerate(lines[:20], 1):
                print(f"  {i:2}: {line.rstrip()}")
    
    def run_module(self, module_num, action="exercise"):
        """
        Run exercises for a module.
        
        Args:
            module_num (int): Module number (1-5)
            action (str): What to run
        """
        if module_num not in self.modules:
            print(f"Error: Invalid module number. Choose 1-5")
            return
        
        print(f"\n{'=' * 70}")
        print(f"MODULE {module_num}: {self.modules[module_num]}")
        print(f"{'=' * 70}")
        
        if action == "exercise":
            self.run_file(self.get_file_path(module_num, "exercise"))
        
        elif action == "solution":
            self.run_file(self.get_file_path(module_num, "solution"))
        
        elif action == "both":
            print("\n" + "=" * 70)
            print("RUNNING EXERCISES")
            print("=" * 70)
            self.run_file(self.get_file_path(module_num, "exercise"))
            
            input("\nPress Enter to see solutions...")
            
            print("\n" + "=" * 70)
            print("RUNNING SOLUTIONS")
            print("=" * 70)
            self.run_file(self.get_file_path(module_num, "solution"))
        
        elif action == "compare":
            self.compare_files(module_num)
        
        elif action == "info":
            self.list_module_contents(module_num)
        
        else:
            print(f"Unknown action: {action}")
            print("Use: exercise, solution, both, compare, or info")


def main():
    """Main entry point."""
    runner = ExerciseRunner()
    
    # No arguments - show menu
    if len(sys.argv) == 1:
        runner.show_menu()
        return
    
    # Parse arguments
    try:
        module_num = int(sys.argv[1])
    except (ValueError, IndexError):
        runner.show_menu()
        return
    
    action = sys.argv[2] if len(sys.argv) > 2 else "exercise"
    
    # Validate module number
    if module_num < 1 or module_num > 5:
        print(f"Error: Module must be 1-5, got {module_num}")
        runner.show_menu()
        return
    
    # Run the module
    runner.run_module(module_num, action)


if __name__ == "__main__":
    main()
