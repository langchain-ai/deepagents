#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. According to Lee’s Market’s internal IT tracking for Q3 2025, inventory system access requests make up 35% of monthly service desk emails. Knowing this highlights where catalog consolidation or workflow improvements could reduce repeated emails and speed resolution.

2. According to internal tracking, shrimp crisp products generated 47 confusing emails in Q3 2025. Knowing this identifies catalog items where clearer organization or guidance could reduce service desk email volume.

3. According to Lee’s Market IT tracking in October 2025, service desk emails peak Monday mornings and Friday afternoons, especially within two hours after system updates. Identifying these spikes helps plan workflow improvements to reduce overall email volume.' > /app/report.md
