# Trace analysis

## Tools

| tool | calls |
| --- | ---: |
| `execute` | 117 |
| `read_file` | 32 |

Chaining: 294 browse verbs across 111 shell calls = 2.65 verbs per model round-trip.

## Where the steps go

| browse verb | calls | share |
| --- | ---: | ---: |
| `get markdown` | 81 | 27.6% |
| `web-open` | 81 | 27.6% |
| `wait timeout` | 29 | 9.9% |
| `wait load` | 16 | 5.4% |
| `eval` | 16 | 5.4% |
| `get url` | 15 | 5.1% |
| `snapshot` | 15 | 5.1% |
| `click` | 13 | 4.4% |
| `get title` | 5 | 1.7% |
| `stop` | 5 | 1.7% |
| `screenshot` | 4 | 1.4% |
| `type` | 4 | 1.4% |
| `press` | 3 | 1.0% |
| `status` | 2 | 0.7% |
| `wait selector` | 2 | 0.7% |
| `back` | 1 | 0.3% |
| `get text` | 1 | 0.3% |
| `fill` | 1 | 0.3% |

## Per task

| task | grade | calls | chain | out chars | big reads | max repeat | tokens | $ | s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Google Search--14 | FAILURE | 24 | 2.78 | 74,454 | 0 | 2 | 871,439 | 0.291 | 481.7 |
| Google Flights--27 | FAILURE | 13 | 3.11 | 43,410 | 0 | 2 | 228,107 | 0.089 | 235.4 |
| Google Flights--3 | TIMEOUT | 11 | 3.0 | 83,587 | 2 | 1 | 350,799 | 0.139 | 902.6 |
| Booking--32 | FAILURE | 10 | 3.25 | 92,248 | 1 | 1 | 188,349 | 0.123 | 205.0 |
| Amazon--25 | SUCCESS | 9 | 1.88 | 14,556 | 0 | 3 | 119,260 | 0.054 | 359.3 |
| Apple--4 | SUCCESS | 6 | 2.2 | 21,037 | 0 | 4 | 81,061 | 0.045 | 103.2 |
| BBC News--37 | SUCCESS | 6 | 2.4 | 60,325 | 1 | 2 | 169,158 | 0.100 | 185.3 |
| Allrecipes--9 | BLOCKED | 4 | 3.0 | 8,111 | 0 | 2 | 45,587 | 0.020 | 72.4 |
| Apple--34 | SUCCESS | 4 | 1.67 | 37,231 | 1 | 1 | 88,186 | 0.064 | 82.8 |
| ArXiv--23 | SUCCESS | 4 | 2.0 | 29,776 | 0 | 1 | 47,792 | 0.047 | 185.3 |
| Coursera--26 | SUCCESS | 4 | 3.0 | 64,399 | 2 | 1 | 108,635 | 0.092 | 163.9 |
| Google Search--40 | BLOCKED | 4 | 3.33 | 12,517 | 0 | 1 | 49,586 | 0.026 | 138.0 |
| Huggingface--37 | SUCCESS | 4 | 1.67 | 58,987 | 1 | 1 | 117,955 | 0.094 | 163.5 |
| Wolfram Alpha--3 | SUCCESS | 4 | 1.67 | 9,499 | 0 | 1 | 37,539 | 0.021 | 132.9 |
| Wolfram Alpha--36 | SUCCESS | 4 | 2.0 | 8,304 | 0 | 1 | 49,600 | 0.026 | 143.2 |
| Allrecipes--20 | BLOCKED | 3 | 4.0 | 7,758 | 0 | 2 | 35,315 | 0.017 | 62.2 |
| Amazon--3 | SUCCESS | 3 | 2.0 | 9,568 | 0 | 1 | 37,060 | 0.022 | 134.1 |
| ArXiv--6 | SUCCESS | 3 | 2.0 | 16,481 | 0 | 1 | 43,327 | 0.028 | 134.0 |
| Coursera--27 | SUCCESS | 3 | 3.0 | 59,305 | 2 | 1 | 71,303 | 0.081 | 123.2 |
| ESPN--4 | SUCCESS | 3 | 3.0 | 23,536 | 0 | 1 | 35,349 | 0.043 | 143.4 |
| ESPN--15 | SUCCESS | 3 | 3.0 | 14,673 | 0 | 1 | 38,971 | 0.028 | 163.9 |
| GitHub--5 | SUCCESS | 3 | 2.5 | 23,156 | 0 | 1 | 42,278 | 0.035 | 163.9 |
| GitHub--35 | FAILURE | 3 | 3.5 | 54,229 | 1 | 1 | 54,741 | 0.066 | 174.2 |
| BBC News--3 | SUCCESS | 2 | 2.0 | 33,204 | 1 | 1 | 37,983 | 0.040 | 62.3 |
| Booking--13 | FAILURE | 2 | 3.0 | 14,340 | 0 | 1 | 30,106 | 0.027 | 102.6 |
| Cambridge Dictionary--2 | SUCCESS | 2 | 2.0 | 38,095 | 1 | 1 | 40,710 | 0.052 | 123.2 |
| Cambridge Dictionary--5 | SUCCESS | 2 | 2.0 | 37,270 | 1 | 1 | 39,590 | 0.049 | 112.9 |
| Google Map--36 | SUCCESS | 2 | 3.0 | 12,870 | 0 | 1 | 29,725 | 0.025 | 143.4 |
| Google Map--7 | SUCCESS | 2 | 3.0 | 9,918 | 0 | 1 | 27,176 | 0.018 | 133.2 |
| Huggingface--40 | SUCCESS | 2 | 2.0 | 12,055 | 0 | 1 | 27,831 | 0.021 | 122.9 |

**Totals.** 30 tasks, 149 tool calls, 3,144,518 tokens, $1.78. Median 4 calls / 47,792 tokens per task.

## Repeated commands (issued 3+ times in one thread)

_none_
