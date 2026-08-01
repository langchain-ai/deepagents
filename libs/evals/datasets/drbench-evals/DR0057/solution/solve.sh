#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. In September 2021, MediConn Solutions instituted a retention playbook that includes semi-annual account reviews, automated messaging for expiring users, and dedicated client success managers, reducing voluntary churn by 8% in the first half of 2022.

2. In Q1 2023 Julian Lee implemented quarterly Net Promoter Score (NPS) surveys for clients who converted to full-time users, tracked satisfaction trends, and created monthly action plans based on specific negative feedback, leading to a 10% improvement in renewal rates over 6 months.' > /app/report.md
