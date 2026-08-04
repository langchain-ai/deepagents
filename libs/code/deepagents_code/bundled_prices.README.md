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
- Remove each entry as soon as upstream's `data.json` covers the model. The
  hourly auto-update picks that up well before the release that would bump our
  pin, so an entry usually goes inert on merge rather than on release. The
  override only fires on a primary-catalog miss, so a stale entry is inert
  rather than harmful — but dead entries still cost review time.
- Do not use this file to override rates for models upstream already prices.
  Upstream always wins: the override catalog is never consulted when the
  primary lookup succeeds.

Users can add their own overrides for models neither catalog covers via
`prices.json` in the dcode user config directory (`~/.deepagents/prices.json`,
same provider-array schema). On conflicting `(provider id, model id)` entries,
the user file wins over this built-in one.
