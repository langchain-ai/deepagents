# Built-in pricing overrides

`bundled_prices.json` is the maintainer-curated pricing catalog `cost_tracking`
consults when the active genai-prices catalog — the bundled data, or the
auto-updated snapshot once one is installed — has no rates for a model. It
exists for one situation: a model users already run has shipped, but upstream
does not price it yet.

The file uses the raw provider-array schema of genai-prices'
`prices/new_data/v2/data.json`, so entries are copy-pasteable into an upstream
PR. JSON has no comments; this policy lives here instead:

- Every entry must be backed by an upstream genai-prices PR (or issue) opened
  first, and must carry a `price_comments` field linking it (e.g.
  `"Stopgap pending pydantic/genai-prices#123"`). An entry without a tracked
  upstream path is one nobody will remember to remove.
  `test_every_bundled_override_entry_is_priced_and_links_upstream` enforces
  this, so a missing link fails the suite rather than the review.
- Remove each entry as soon as upstream's `data.json` covers the model. The
  hourly auto-update picks that up well before the release that would bump our
  pin, so an entry usually goes inert on merge rather than on release. The
  override only fires on a primary-catalog miss, so a stale entry is normally
  inert rather than harmful — but that depends on the primary lookup actually
  succeeding, which is not guaranteed when the provider id LangChain reports
  differs from the one upstream cataloged. In that case a stale entry keeps
  billing its own possibly-outdated rate, silently. Dead entries also cost
  review time.
- Do not use this file to override rates for models upstream already prices.
  Upstream always wins: the override catalog is never consulted when the
  primary lookup succeeds.

## Adding an entry

The file ships as an empty array (`[]`), so there is no in-file example. A
minimal one entry:

```json
[
  {
    "id": "anthropic",
    "name": "Anthropic",
    "api_pattern": "api\\.anthropic\\.com",
    "models": [
      {
        "id": "claude-example-5",
        "match": { "equals": "claude-example-5" },
        "price_comments": "Stopgap pending pydantic/genai-prices#123",
        "prices": { "input_mtok": 3.0, "output_mtok": 15.0 }
      }
    ]
  }
]
```

Required fields, none of which the schema will fill in for you: `id`, `name`,
and `api_pattern` on the provider; `id`, `match`, and `prices` on each model.
`price_comments` exists on both types — put it on the **model**, since that is
what a reviewer needs to trace and what the policy test reads (it falls back to
the provider's).

Getting any of this wrong is quiet: `_build_price_overrides` logs one warning
and drops the source, and the model then shows `$0` — exactly what it showed
before the entry was added. The `provider.id` must be the id genai-prices uses,
because that is what dcode's provider aliasing resolves to.

Beyond `input_mtok` / `output_mtok`, the schema carries `cache_read_mtok`,
`cache_write_mtok`, `output_reasoning_mtok`, `input_audio_mtok`, and tiered
variants. Only publish a bucket you actually have a rate for: tokens in an
omitted bucket stay in the ordinary input or output total rather than being
priced separately.

## User overrides

Users can add their own overrides for models neither catalog covers via
`prices.json` in the dcode user config directory (`~/.deepagents/prices.json`,
same provider-array schema). On conflicting `(provider id, model id)` entries,
the user file wins over this built-in one. See
[`PRICING.md`](https://github.com/langchain-ai/deepagents/blob/main/libs/code/PRICING.md)
for the user-facing documentation — linked by URL because this file ships inside
the wheel, where the repo tree is not there to walk.
