# Model pool for the Switchyard routing benchmark. The ONLY place model ids live.
# Edit here, run ./render.sh, and all six route configs regenerate.
#
#   source models.sh && ./render.sh
#
# Arms:
#   Baselines: glm, opus, nano (passthrough)
#   Escalation: escalation (Opus↑GLM), glm-nano, opus-nano
#
# All six run THROUGH Switchyard so every arm is measured by the same
# instrument (/v1/stats). Running the baselines direct-to-provider would give
# no per-model cache read/write split, and cost would not be comparable.

# ---- weak tier: GLM 5.2 on Baseten (OpenAI-compatible) ----
# Not in this repo's Baseten registry (.github/scripts/evals/models.py lists
# MiniMax-M2.5, Kimi-K2.6, Nemotron-120B-A12B, Qwen3-Coder-480B), so the id and
# endpoint come from Baseten's own model page rather than from the registry.
#
# Pricing caveat: genai-prices matches "zai-org/GLM-5.2" on the model id and
# returns the same rate for every provider, so the absolute $/run is a list-price
# estimate that may not equal Baseten's invoice. Cross-arm comparison is
# unaffected — every arm prices GLM off the same entry.
# Direct to Baseten, NOT the LangSmith gateway. Verified: the gateway rejects
# Baseten with "BASETEN_API_KEY not found: secret not found in workspace
# secrets", while inference.baseten.co accepts BASETEN_API_KEY directly.
# Hardcoded on purpose — do not fall back to $BASETEN_BASE_URL. That variable is
# set to the gateway in a normal dev shell, so inheriting it silently points
# this route somewhere that 403s.
export SY_WEAK_ID="zai-org/GLM-5.2"
export SY_WEAK_BASE_URL="https://inference.baseten.co/v1"
export SY_WEAK_KEY_ENV="BASETEN_API_KEY"

# ---- alternate weak tier: Nemotron 3.5 Nano ("Lightning") on NVIDIA NIM ----
# Preview/private model, so it is not in any public price catalog. Verified
# working: tool calling succeeds in both thinking modes.
#
# SY_NANO_THINKING defaults ON, matching how NVIDIA ships and demonstrates the
# model. Benchmarking it with thinking disabled would measure a configuration
# nobody runs and would understate the model in a joint artifact. It also
# matches this repo's existing convention of evaluating models at their
# intended reasoning setting (see _apply_glm_5_2_reasoning_default).
#
# Cost impact is affordable: output is ~2% of token volume but 25% of spend on
# this suite, and the realistic thinking multiplier here is ~1.5-3x (measured
# 1.5x on a tool call, which is what most of these tests are) rather than the
# 30x seen on a degenerate one-line question. That lands around 1.25-1.5x total
# cost. Flip to false only to isolate the cost of reasoning as a variant.
export SY_NANO_ID="private/nvidia/nemotron-3.5-nano-30b-a3b"
export SY_NANO_BASE_URL="https://integrate.api.nvidia.com/v1"
export SY_NANO_KEY_ENV="NVIDIA_API_KEY"
export SY_NANO_THINKING="true"

# ---- alternate strong tier: Nemotron 3 Ultra on Baseten ----
# Verified live on the Baseten catalog. Caches like GLM does there, which is
# what makes it cheaper than Nano per task despite being a 550B model.
export SY_ULTRA_ID="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"
export SY_ULTRA_BASE_URL="https://inference.baseten.co/v1"
export SY_ULTRA_KEY_ENV="BASETEN_API_KEY"

# ---- strong tier: Opus 4.8, native Anthropic ----
# Native Anthropic Messages format. Switchyard translates the incoming
# OpenAI-Chat request to it, and marks the final content block as an ephemeral
# cache breakpoint (libsy-llm-client/src/client.rs:707), so Opus still gets
# prompt-cache hits through the proxy.
# Through the LangSmith gateway, because ANTHROPIC_API_KEY in this environment
# holds a gateway key (lsv2_...) rather than an sk-ant key; a gateway-only key
# 403s against api.anthropic.com. Verified the gateway accepts Bearer auth and
# passes through cache_read_input_tokens / cache_creation_input_tokens, which
# the cost split depends on.
export SY_STRONG_ID="claude-opus-4-8"
export SY_STRONG_BASE_URL="https://gateway.smith.langchain.com/anthropic"
export SY_STRONG_KEY_ENV="ANTHROPIC_API_KEY"

# ---- judge: small, fast, and on its own quota bucket ----
# Matches Sean's runs. Must NOT be a Claude/Bedrock model: those reject the
# chat_template_kwargs hint outright, every judged turn fails open to weak, and
# the router silently never escalates.
# Also via the gateway (GOOGLE_API_KEY is likewise an lsv2_ gateway key). Only
# /gemini/v1/chat/completions is allow-listed there — the /v1beta/openai and
# /openai/v1 paths both return 501 "path not allow-listed by gateway".
export SY_JUDGE_ID="gemini-3.1-flash-lite"
export SY_JUDGE_BASE_URL="https://gateway.smith.langchain.com/gemini/v1"
export SY_JUDGE_KEY_ENV="GOOGLE_API_KEY"

# ---- escalation tuning ----
# confirmations is the main cost dial. 2 is the benchmarked default and Sean's
# setting: it latches ~1/3 as often as 1. NOTE: >1 requires a session id, which
# conftest.py supplies per test via the x-switchyard-session-id header.
export SY_CONFIRMATIONS="2"
export SY_RECENT_TURN_WINDOW="28"
export SY_WINDOW_MESSAGE_CHARS="500"
