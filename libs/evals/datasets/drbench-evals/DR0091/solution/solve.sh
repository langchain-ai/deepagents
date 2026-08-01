#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. In Q4 2025, Lee'"'"'s Market'"'"'s average time agents spent gathering basic, repeated customer context was 1.8 minutes per call, representing 17% of the average call duration for escalated tickets.

2. At Lee'"'"'s Market, the FCR rate for customers who authenticated and used the personalized self-service portal (referencing their last 3 interactions) was 88% in H2 2025, which is 12 percentage points higher than the average FCR for all channels combined.

3. Customers who received at least one personalized follow-up email referencing their specific product model or issue in FY 2025 have an average Customer Lifetime Value (LTV) that is 34% higher than those who only received standard, generic templates.

4. Accounts identified by the system as having a service issue but who received a proactive status alert via text/app notification registered a monthly churn rate of 0.9% in Q3 2025, which is 33% lower than the 1.35% rate for accounts that waited until the issue was manually reported.

5. Lee'"'"'s Market'"'"'s average conversion rate for upsell/cross-sell offers made within the exclusive "Fast-Tracked" live chat channel in Q4 2025 was 15.2%, generating 2.5 times the Average Revenue Per User (ARPU) compared to transactions originating from the standard support queue.

6. 65% of customers who had their issue resolved in under 5 minutes via chat or phone in 2025 rated the experience a 5/5 (Very Satisfied).' > /app/report.md
