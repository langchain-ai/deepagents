# Apple TV processor lookup

## What this measures

Whether the browsing agent reads a spec off a live product page rather than
answering from model memory. The answer is widely known, so a correct response
is only evidence of browsing if the agent actually reached the page -- the
trajectory matters as much as the string.

Adapted from WebVoyager (`Apple--25`).

## Success

The final answer names the processor in the current Apple TV 4K: the A15 Bionic.
The verifier accepts the chip name in any casing, with or without the "Bionic"
suffix, since the agent's phrasing varies.

## Why a keyword verifier, not an LLM judge

An LLM judge flipped identical result sets between runs earlier in this project,
which made small differences between harnesses unreadable. This question has one
factual answer, so a deterministic check removes the grader as a source of noise.

The tradeoff is honest: the keyword check cannot tell a page-sourced answer from
a recalled one. It measures "said the right thing", not "found it on the page".
Pair it with the trajectory when that distinction matters.

## Environment note

Harbor runs the project's own `sandbox/setup.sh` inside the trial image, so the
image has to satisfy the agent's sandbox dependencies (Node for the `browse`
install) as well as the MDA adapter's runtime install, which shells out to
`python` and `python -m pip`.
