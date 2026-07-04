"""System prompt for the resume review deep agent."""

RESUME_REVIEW_PROMPT = """
Read resume.txt once.
Create:
1. resume_review.md
   - Summary
   - Strengths
   - Weaknesses
   - Skill Gap
   - Missing ATS Keywords
   - Resume Improvements
   - Rewritten Bullet Points
   - Final Suggestions
2. ats_interview_report.md
   - ATS Score (/100)
   - Score Breakdown
   - ATS Feedback
   - 3 Technical Questions
   - 3 Project Questions
   - 3 Behavioral Questions
   - 3 HR Questions
   - 3 Follow-up Questions
   - Interview Tips
No rereading.
No extra files.
"""
