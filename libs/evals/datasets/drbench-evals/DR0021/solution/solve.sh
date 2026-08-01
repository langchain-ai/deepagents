#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. According to Lee’s Market’s ITSM tracker for Q3 2024, average incident resolution time increased by 22%. The increase in time is caused by regional stores using separate IT systems, and regional teams being unable to communicate consistently. Also, processes for managing incidents, problems, and changes are not standardized.

2. According to internal IT reports from June 2024, employees spent an average of 3.5 hours per week manually tracking requests because no automated workflow exists to prioritize or assign tasks, causing inefficiencies across regions.

3. According to Lee’s Market’s IT operations'"'"' logs from September 2024, rolling out updates to all stores takes an average of 5 business days because each location uses slightly different systems, requiring updates to be applied individually, which delays completion across regions.' > /app/report.md
