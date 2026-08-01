#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. The percentage dip in average customer dwell time on the sales floor between May and July 2025 was 18%.

2. The percentage dip in the number of product interactions per customer on the sales floor between May and July 2025 was 40%.

3. Eight of the top 10 seasonal SKUs underperformed from June to August 2025 compared to the same period in 2024.

4. The average order size of Asian customers was 18% higher than the average order size of non-Asian customers during July 2025.

5. The average basket value for customers who interacted with sales associates was 22% higher than for independent shoppers.

6. 35% of customers who received personalized recommendations returned for additional purchases in August 2025.' > /app/report.md
