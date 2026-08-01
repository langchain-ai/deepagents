#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. In 2024, healthcare cyberattacks remain at record levels, with 386 attacks reported in the year'"'"'s first part, matching the pace of 2023, the worst year for breaches. These breaches have significantly impacted both data security and clinical operations.

2. Ransomware attacks in healthcare pose "threat-to-life" risks by shutting down critical systems and delaying patient care. A February 2024 attack on Change Healthcare, a major third-party health IT provider, disrupted nationwide patient care and halted billions in payments, highlighting the severe impact a single cyber incident can have on healthcare delivery.

3. Healthcare'"'"'s third-party data breaches surged by 287% from 2022 to 2023, affecting a significant number of individuals. The evolving cybersecurity threats, marked by collaboration between nation-state hackers and ransomware groups, have led to calls for stronger coordinated defenses, including new HHS cybersecurity guidelines, to safeguard patient data and maintain service continuity.

4. 25% of our virtual healthcare platforms have been compromised in the last 12 months, resulting in 15 significant data breaches.

5. Data breach costs totaled $3 million in Q4 2024.

6. Eight breaches involved third parties in Q2 2025.

7. Third-party compliance rate was 55% in Q1 2025.

8. Only 45% of third-party vendors achieved compliance by Q3 2025.

9. Data breaches in Q1 2024 cost MediConn approximately $2 million.

10. The average cost of a data breach to MediConn in the last 2 years is $450,000, with 80% of those breaches occurring on third-party servers.' > /app/report.md
