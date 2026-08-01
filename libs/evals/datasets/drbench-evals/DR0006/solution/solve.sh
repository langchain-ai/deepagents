#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. Telehealth providers must choose between aligning with insurance, which requires full HIPAA compliance, or operating on a cash-only basis, thereby avoiding HIPAA but navigating a complex landscape of 50-state privacy laws.

2. Accepting insurance expands the patient base and ensures steady reimbursements, but necessitates significant investment in secure platforms to comply with HIPAA Privacy and Security standards, including data encryption, risk analyses, and comprehensive staff training.

3. Cash-only telehealth provides flexibility with provider-set rates and simpler billing but poses legal risks due to complex state-specific data privacy regulations. New state health data laws allow consumers to sue over privacy breaches, unlike HIPAA, which offers protections for providers.

4. 5 violations reported in Q3 2025.

5. Projected HIPAA compliance cost is $2.5M for Q2 2024.

6. 60% of providers accept insurance by Q3 2025.

7. Staff training requires 1,200 hours.' > /app/report.md
