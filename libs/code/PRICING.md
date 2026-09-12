# Cost estimates and local pricing overrides

`/cost` reports an estimate for the current thread. Rates come from
[genai-prices](https://github.com/pydantic/genai-prices), which ships an offline
catalog and refreshes it hourly in the background (opt out with
`DEEPAGENTS_CODE_PRICES_AUTO_UPDATE=0`, or `[update].prices_auto_update = false`
in `config.toml`).

Estimates are display-only. Nothing in dcode caps spend or gates execution on
them, so a wrong rate costs you an inaccurate number and nothing else.

## When a model has no rates

A model the catalog does not cover is left out of the total. `/cost` distinguishes
that from a genuinely free request: it reports how many of the recorded requests
are included in the figure.

That happens with a newly released model, a self-hosted or proxied endpoint, or a
provider name that does not line up with what genai-prices calls it. If you would
rather see an estimate than a gap, write your own rates to:

```
~/.deepagents/prices.json
```

The file is read once, on the first request that needs it. **Edits take effect on
the next dcode start**, not mid-session.

## File format

`prices.json` uses the same provider-array schema as genai-prices' own
`prices/new_data/v2/data.json`, so an entry can be contributed upstream as-is.
It is a JSON array of providers:

```json
[
  {
    "id": "my-proxy",
    "name": "My proxy",
    "api_pattern": "gateway\\.example\\.internal",
    "models": [
      {
        "id": "house-model-v2",
        "match": { "equals": "house-model-v2" },
        "prices": { "input_mtok": 2.5, "output_mtok": 10.0 }
      }
    ]
  }
]
```

Required, and easy to omit:

| Field | Where | Notes |
| --- | --- | --- |
| `id` | provider | Must be the provider id dcode reports for the request — see below. |
| `name` | provider | Any display string. |
| `api_pattern` | provider | Regex against the API URL. Unused by dcode, but the schema requires it. |
| `id` | model | Any identifier; used in log messages. |
| `match` | model | How to recognize the model: `{"equals": ...}`, `{"starts_with": ...}`, `{"contains": ...}`, or `{"regex": ...}`. |
| `prices` | model | At least one rate. |

Rates are per million tokens: `input_mtok`, `output_mtok`, and optionally
`cache_read_mtok`, `cache_write_mtok`, `output_reasoning_mtok`,
`input_audio_mtok`. Only list a bucket you have a real rate for — tokens in an
omitted bucket are billed at the ordinary input or output rate rather than being
dropped.

## Getting the provider id right

This is where a hand-written file usually goes wrong. dcode resolves the
LangChain provider name through its own alias table before looking anything up,
so the id it searches on is not always the string you would expect.

If no provider in your file claims that id, dcode falls back to searching **every**
provider in the file by model id alone. That keeps your entry reachable, but it
also means an entry can price a request that ran against a different provider —
`llama-3.3-70b` costs very different amounts on Bedrock, Together, and Groq. When
that happens you get a warning naming both providers; pin the entry by using the
id from the warning.

## Diagnosing a file that is not working

Everything below is logged to the Debug Console (`Ctrl+\`), each message once per
session:

| What you see | What it means |
| --- | --- |
| Nothing at all, cost still missing | The file is not where dcode looks, or is empty. Both are silent by design. |
| "No pricing override matched model=… though … contributed N model entries" | The file loaded, but nothing matched. The `provider_id` in the message is the post-alias id your entry must use. |
| "Could not parse … as JSON" / "must be an array of providers" | Syntax or shape problem. The payload must be a bare array, not `{"providers": [...]}`. |
| "the entries failed validation against the genai-prices provider schema" | A required field is missing or mistyped. |
| "the request ran against provider …" | Priced by a sweep, possibly from the wrong provider's entry. See above. |
| "its entry matched but its rates were rejected" | A rate is unusable — negative, or large enough to overflow. |

A malformed `prices.json` never breaks a model turn and never disables ordinary
pricing: the bad source is dropped and everything else still prices.

## Built-in stopgaps

dcode also ships a small maintainer-curated catalog for models upstream has not
priced yet. Your file wins over it on a conflicting `(provider id, model id)`
pair, and both are consulted only when genai-prices itself has no rates. See
[`deepagents_code/bundled_prices.README.md`](./deepagents_code/bundled_prices.README.md)
for that policy.
