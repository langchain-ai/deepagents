#!/bin/sh
set -eu
printf '%s\n' '# Reference report

1. Elexion internal planning data for Q1 2026 shows lithium-ion batteries account for 80% of planned EV allocations, solid-state batteries 15%, and sodium-ion batteries 5%.

2. Elexion'"'"'s internal Supplier Reports indicate that combined production capacity is projected to meet only 85% of projected EV demand for Q2 2026, with potential shortfalls concentrated in lithium-ion supply.

3. Elexion internal logistics projections for Q2 2026 indicate that delivery delays will be highest in the Northeast US, where shipments are expected to face delays of up to 12 days due to port congestion and limited local battery inventory.

4. Elexion internal R&D tracking shows that 22% of prototypes are equipped with solid-state batteries and 8% with sodium-ion batteries, while the remaining 70% still use lithium-ion.

5. As of October 2025, Elexion internal logistics tracking indicates that an average of 78% of planned battery inventory is in stock across all distribution centers, with the lowest levels at the Midwest and Northeast hubs.' > /app/report.md
