#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. According to Lee’s Market internal tracking from Q2 2025, employees across multiple stores submitted an average of 480 duplicate IT requests every week, primarily for password resets and printer errors, due to a confusing self-service portal.

2. According to Lee’s Market internal logs from Q3 2025, requests related to online ordering system errors take the longest to resolve, averaging 2.5 hours per incident, due to a lack of clear self-service guidance.

3. According to Lee’s Market internal IT performance reports from Q3 2025, service desk requests spike by 40% on Mondays and Fridays at around 11 a.m., with average resolution times increasing by up to 3 hours when self-service guidance is unclear.' > /app/report.md
