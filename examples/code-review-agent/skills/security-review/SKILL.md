---
name: security-review
description: For security-focused code review targeting vulnerabilities like injection, authentication issues, secrets exposure, and unsafe operations
---

# Security Review Skill

A structured workflow for security-focused code review, aligned with OWASP guidelines.

## When to Use This Skill

Use this skill when asked to:
- Perform a security review or security audit
- Check code for vulnerabilities before deployment
- Review authentication, authorization, or input handling code
- Assess code that handles sensitive data

## Workflow

### 1. Identify the Stack

Determine the language, framework, and dependencies:
```
ls()
glob("**/requirements*.txt")      # Python
glob("**/package.json")           # JavaScript/TypeScript
glob("**/go.mod")                 # Go
glob("**/Cargo.toml")             # Rust
glob("**/pom.xml")                # Java
```

Read dependency files to check for known-vulnerable versions.

### 2. Scan for Security-Sensitive Patterns

Run targeted `grep` searches for each vulnerability class:

**Injection (SQL, Command, Template):**
```
grep("execute\\(|cursor\\.|raw\\(|rawQuery")     # SQL
grep("subprocess|os\\.system|os\\.popen|exec\\(")  # Command injection
grep("eval\\(|Function\\(|compile\\(")            # Code injection
grep("render_template_string|Markup\\(|safe\\|")  # Template injection
```

**Cross-Site Scripting (XSS):**
```
grep("innerHTML|outerHTML|document\\.write")
grep("dangerouslySetInnerHTML")
grep("v-html|\\[innerHTML\\]")
```

**Secrets and Credentials:**
```
grep("password|passwd|secret|api_key|apikey|token|credential")
grep("BEGIN.*PRIVATE KEY|BEGIN.*RSA")
grep("AKIA[0-9A-Z]")                              # AWS access keys
```

**Insecure Cryptography:**
```
grep("md5|sha1|DES|RC4")                           # Weak algorithms
grep("random\\(\\)|Math\\.random|rand\\(\\)")     # Weak randomness for security
```

**Path Traversal:**
```
grep("\\.\\./|path\\.join.*\\+|open\\(.*\\+")
```

**Insecure Configuration:**
```
grep("CORS|Access-Control|allowOrigin")
grep("debug.*=.*[Tt]rue|DEBUG.*=.*1")
grep("http://")                                     # Non-HTTPS URLs
```

### 3. Deep-Read Flagged Files

For each file with matches, use `read_file` to understand the full context:
- Is the pattern actually exploitable?
- Is there input validation upstream?
- Is the code in a test file (lower severity)?
- Is there a mitigating control?

### 4. Delegate for Deep Analysis

For complex findings, delegate to the `security-analyzer` subagent:

```
task(
    subagent_type="security-analyzer",
    description="Analyze these files for [specific concern]: [file1], [file2]. I found [pattern] that may indicate [vulnerability]."
)
```

### 5. Write the Report

Include findings categorized by vulnerability type:
- Injection
- Authentication / Authorization
- Sensitive Data Exposure
- Security Misconfiguration
- Insecure Dependencies

## Severity Guide

| Severity | Criteria | Example |
|----------|----------|---------|
| CRITICAL | Exploitable with direct impact | SQL injection in user input handler |
| WARNING | Potentially exploitable or risky pattern | Hardcoded API key in source |
| SUGGESTION | Defense-in-depth improvement | Adding Content-Security-Policy header |

## Quality Checklist

Before finishing:
- [ ] All vulnerability categories have been scanned with grep
- [ ] Flagged files have been read in full context
- [ ] Findings distinguish between actual vulnerabilities and false positives
- [ ] Each finding has severity, file reference, and remediation
- [ ] Dependency files have been checked
- [ ] Report is saved to the specified output file
