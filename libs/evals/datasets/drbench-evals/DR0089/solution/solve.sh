#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. The temperatures at which Elexion Automotive'"'"'s current weather testing protocols validate EV performance are 20°C and -7°C.

2. According to Elexion'"'"'s Annual Testing Report for 2024, cold-weather testing should be conducted at -20°C, -15°C, -10°C, -5°C, and 0°C to ensure better endurance coverage.

3. 70% of Elexion Automotive'"'"'s manufacturing plants isolate EV battery pack tests from HVAC or cabin heating simulations.

4. The percentage point gap between EV test cases that include preconditioning cycles and the real-world usage statistics of such cases in cold weather conditions is 20% at Elexion Automotive. 
Only 40% of EV test cases include preconditioning cycles at Elexion Automotive, although real drivers use them in approximately 60% of cold starts.

5. 6-8 full-vehicle cold chambers are recommended to be added by Q4 2026 to expand testing capacity at Elexion Automotive.

6. Integrated protocol models at Elexion Automotive retained 9 to 11 percent more winter range over six months.

7. The EV cold-weather testing plan at Elexion Automotive from Q4 2025 to Q4 2026 includes three phases. Phase A, scheduled for Q4 2025, focuses on conducting a gap audit and initiating pilot testing. Phase B, spanning Q1 to Q2 2026, aims to expand both laboratory and system-level tests. Phase C, planned for Q3 to Q4 2026, focuses on production validation and ongoing reliability monitoring, ensuring a comprehensive evaluation of EV performance in cold weather conditions.' > /app/report.md
