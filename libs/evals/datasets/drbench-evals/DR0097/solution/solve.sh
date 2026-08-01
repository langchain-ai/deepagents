#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. MediConn'"'"'s IT cost per consult is approximately $10 CAD as of October 2025.

2. 40% of clinician worktime per visit is spent on direct patient interaction and diagnosis.

3. EHR documentation, data entry, and navigating across platforms for calls, reports, and expert opinions consume approximately 50% of the clinician work time.

4. 50% of EHR and documentation workflows are automated as of October 2025.

5. MediConn clinicians currently use three specialized platforms: one for patient interaction, one for lab results, and one for expert consultations.

6. The AI scribing pilot automatically generated visit summaries and completed EHR documentation, resulting in a 70% reduction in clinician time spent on these tasks compared to previous levels.

7. The UC Platform combined messaging, video consultations, lab results review, and expert opinion requests into a single interface. This eliminated time spent juggling between systems and improved navigation speed by 90% compared to previous workflows.

8. The automated lab result annotation pilot, designed to reduce clinician review time by automatically flagging abnormal values and generating preliminary notes, instead led to a 20% increase in documentation errors and added workflow complexity.' > /app/report.md
