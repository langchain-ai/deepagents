"""
FINAL PROJECT 2: PASSWORD STRENGTH CHECKER
===========================================

Build a comprehensive password validation and strength checker.

Features:
- Check password strength with detailed scoring
- Validate password requirements
- Suggest improvements
- Track password history (in memory)
- Interactive and demo modes
- Educational comments

This project demonstrates:
- String manipulation and validation
- Functions with multiple returns
- Data structures (lists, dicts)
- Exception handling
- OOP design
- User interaction
"""

import re
from datetime import datetime


class PasswordValidator:
    """Validates passwords against security criteria."""
    
    # Password requirements
    MIN_LENGTH = 8
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~"
    
    def __init__(self):
        """Initialize validator."""
        self.password_history = []
    
    def validate(self, password):
        """
        Validate password against all criteria.
        
        Returns:
            dict: Validation results with details
        """
        errors = []
        
        # Length check
        if len(password) < self.MIN_LENGTH:
            errors.append(f"Password too short (min {self.MIN_LENGTH} characters)")
        
        # Uppercase check
        if not re.search(r"[A-Z]", password):
            errors.append("Missing uppercase letter (A-Z)")
        
        # Lowercase check
        if not re.search(r"[a-z]", password):
            errors.append("Missing lowercase letter (a-z)")
        
        # Digit check
        if not re.search(r"\d", password):
            errors.append("Missing digit (0-9)")
        
        # Special character check
        if not re.search(f"[{re.escape(self.SPECIAL_CHARS)}]", password):
            errors.append(f"Missing special character {self.SPECIAL_CHARS}")
        
        # Common words check
        common_words = ["password", "123456", "qwerty", "admin", "letmein"]
        if password.lower() in common_words:
            errors.append("Password is too common")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "error_count": len(errors)
        }
    
    def calculate_strength(self, password):
        """
        Calculate password strength score 0-100.
        
        Returns:
            dict: Strength score and level
        """
        score = 0
        
        # Length (up to 30 points)
        length = len(password)
        if length >= self.MIN_LENGTH:
            score += min(30, length * 2)
        
        # Character variety (up to 40 points)
        has_lower = bool(re.search(r"[a-z]", password))
        has_upper = bool(re.search(r"[A-Z]", password))
        has_digit = bool(re.search(r"\d", password))
        has_special = bool(re.search(f"[{re.escape(self.SPECIAL_CHARS)}]", password))
        
        variety_score = sum([has_lower, has_upper, has_digit, has_special]) * 10
        score += variety_score
        
        # Entropy (up to 30 points)
        if len(set(password)) > 6:
            score += 15
        if len(set(password)) > 10:
            score += 15
        
        # Determine level
        if score >= 80:
            level = "Very Strong"
            color = "🟢"
        elif score >= 60:
            level = "Strong"
            color = "🟢"
        elif score >= 40:
            level = "Good"
            color = "🟡"
        elif score >= 20:
            level = "Weak"
            color = "🟠"
        else:
            level = "Very Weak"
            color = "🔴"
        
        return {
            "score": min(100, score),
            "level": level,
            "color": color
        }
    
    def get_suggestions(self, password):
        """
        Get suggestions to improve password.
        
        Returns:
            list: Improvement suggestions
        """
        suggestions = []
        
        if len(password) < 12:
            suggestions.append("Add more characters (aim for 12+)")
        
        if not re.search(r"[A-Z]", password):
            suggestions.append("Add uppercase letters")
        
        if not re.search(r"[a-z]", password):
            suggestions.append("Add lowercase letters")
        
        if not re.search(r"\d", password):
            suggestions.append("Add numbers")
        
        if not re.search(f"[{re.escape(self.SPECIAL_CHARS)}]", password):
            suggestions.append(f"Add special characters: {self.SPECIAL_CHARS}")
        
        # Check for patterns
        if re.search(r"(.)\1{2,}", password):
            suggestions.append("Avoid repeating characters")
        
        if re.search(r"(012|123|234|345|456|567|678|789|890)", password):
            suggestions.append("Avoid sequential numbers")
        
        if re.search(r"(abc|bcd|cde|def)", password):
            suggestions.append("Avoid sequential letters")
        
        return suggestions
    
    def add_to_history(self, password):
        """Track password in history."""
        self.password_history.append({
            "password": password,
            "timestamp": datetime.now().isoformat()
        })
    
    def check_reuse(self, password):
        """Check if password was used before."""
        for entry in self.password_history:
            if entry["password"] == password:
                return True
        return False


class PasswordChecker:
    """Main password checking application."""
    
    def __init__(self):
        """Initialize the checker."""
        self.validator = PasswordValidator()
    
    def analyze_password(self, password):
        """
        Perform complete password analysis.
        
        Returns:
            dict: Complete analysis results
        """
        # Validation
        validation = self.validator.validate(password)
        
        # Strength
        strength = self.validator.calculate_strength(password)
        
        # Suggestions
        suggestions = self.validator.get_suggestions(password)
        
        # Reuse check
        reused = self.validator.check_reuse(password)
        
        return {
            "password_length": len(password),
            "validation": validation,
            "strength": strength,
            "suggestions": suggestions,
            "previously_used": reused
        }
    
    def display_analysis(self, password):
        """Display formatted analysis results."""
        analysis = self.analyze_password(password)
        
        print("\n" + "=" * 60)
        print("PASSWORD STRENGTH ANALYSIS")
        print("=" * 60)
        
        # Length
        print(f"\nLength: {analysis['password_length']} characters", end="")
        if analysis['password_length'] >= 12:
            print(" ✓")
        elif analysis['password_length'] >= 8:
            print(" (good)")
        else:
            print(" (too short!)")
        
        # Strength
        strength = analysis["strength"]
        print(f"\nStrength: {strength['color']} {strength['level']} "
              f"({strength['score']}/100)")
        
        # Validation
        validation = analysis["validation"]
        if validation["valid"]:
            print("\n✓ Password meets all security requirements!")
        else:
            print(f"\n✗ {validation['error_count']} requirement(s) not met:")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        # Reuse
        if analysis["previously_used"]:
            print("\n⚠ Warning: This password was used before!")
        
        # Suggestions
        suggestions = analysis["suggestions"]
        if suggestions:
            print("\n💡 Suggestions to improve:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        
        print("\n" + "=" * 60)
        return analysis


def main():
    """Main interactive mode."""
    checker = PasswordChecker()
    
    print("\n" + "=" * 60)
    print("PASSWORD STRENGTH CHECKER")
    print("=" * 60)
    print("\nCommands: check, history, clear, quit\n")
    
    while True:
        try:
            command = input("> ").strip().lower()
            
            if command == "quit":
                print("Goodbye!")
                break
            
            elif command == "check":
                password = input("Enter password to check: ")
                if password:
                    analysis = checker.display_analysis(password)
                    checker.validator.add_to_history(password)
            
            elif command == "history":
                if checker.validator.password_history:
                    print("\nPassword Check History:")
                    for i, entry in enumerate(checker.validator.password_history, 1):
                        print(f"{i}. {entry['timestamp']}")
                else:
                    print("No history yet.")
            
            elif command == "clear":
                checker.validator.password_history = []
                print("History cleared.")
            
            else:
                print("Unknown command. Try: check, history, clear, quit")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Demonstration mode
    print("\n" + "=" * 60)
    print("PROJECT 2: PASSWORD STRENGTH CHECKER - DEMO")
    print("=" * 60 + "\n")
    
    checker = PasswordChecker()
    
    # Test passwords
    test_passwords = [
        "pass",                      # Too weak
        "password123",               # Common
        "MyPass123",                 # Better
        "MyP@ssw0rd!",              # Strong
        "C0mpl3x!P@ssw0rd#2024",    # Very strong
    ]
    
    print("Testing various passwords:\n")
    
    for pwd in test_passwords:
        print(f"Testing: '{pwd}'")
        analysis = checker.display_analysis(pwd)
        checker.validator.add_to_history(pwd)
        print()
    
    print("=" * 60)
    print("Statistics:")
    print(f"  Passwords checked: {len(checker.validator.password_history)}")
    print(f"  Average score: {sum(a['strength']['score'] for a in [checker.analyzer.check(p) for p in test_passwords if p]):.1f}")
    print("=" * 60)
    
    print("\nTo use interactive mode, uncomment main() call at bottom")
    # Uncomment to run interactive:
    # main()
