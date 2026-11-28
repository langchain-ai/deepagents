#!/usr/bin/env bash
# Prints a JSON array of merged release PRs ({number, title}) that are still
# labeled pending, i.e. whose publish has not tagged its release yet. Prints
# `[]` when none are pending. Exits non-zero when GitHub cannot be read, so a
# caller can tell "nothing pending" apart from "could not tell".
#
# Used by release-please.yml's `guard-pending-release` job and again inside
# the serialized `release-please` job, so both read release state the same way.
#
# Environment:
#   REPO         owner/name (required)
#   INCLUDE_SHA  optional commit SHA. The merged PR for that commit is always
#                checked, whatever the label search returns. Release-commit
#                runs pass their own SHA: the label search goes through the
#                search index, which can lag a fresh merge, and missing your
#                own just-merged release PR is exactly the half-updated state
#                (version bumped, no tag yet) the guard exists to catch.
#
# Both label names are read during migration; the publisher clears both after
# tagging.
set -euo pipefail

: "${REPO:?REPO must be set}"

PENDING_LABELS='["auto:release-pending", "autorelease: pending"]'

# --limit is explicit and generous: the default (30) could drop a genuinely
# stuck pending PR in exactly the backlog scenario the guard exists to catch.
# Separate queries implement OR; multiple --label flags mean AND.
current=$(gh pr list --repo "$REPO" --state merged \
  --label "auto:release-pending" --limit 100 --json number)
legacy=$(gh pr list --repo "$REPO" --state merged \
  --label "autorelease: pending" --limit 100 --json number)

# Type-guarded: a valid-JSON body of the wrong shape (an API error object, an
# HTML error page) must fail rather than read as "nothing pending".
numbers=$(printf '%s\n%s\n' "$current" "$legacy" | jq -se '
  if length != 2 or any(.[]; type != "array")
  then error("expected two arrays of PRs")
  else add | map(.number) end
')

if [ -n "${INCLUDE_SHA:-}" ]; then
  # `/commits/{sha}/pulls` reads the commit's PR association directly, not
  # through the search index. Filter to the merged PR so an unrelated open PR
  # sharing the SHA cannot stand in for it.
  own=$(gh api -H "Accept: application/vnd.github+json" \
    "/repos/${REPO}/commits/${INCLUDE_SHA}/pulls" \
    --jq 'map(select(.merged_at != null)) | .[0].number // empty')
  if [ -z "$own" ]; then
    echo "No merged PR is associated with ${INCLUDE_SHA} yet." >&2
    exit 1
  fi
  numbers=$(jq -c --argjson own "$own" '. + [$own]' <<< "$numbers")
fi

# The label search can also lag a label *removal* by seconds. The refresh
# release.yml dispatches starts right after `mark-release` flips the label, so
# a stale hit would make that refresh defer to a dispatch that never comes.
# Re-read each candidate directly, which is consistent.
confirmed='[]'
for number in $(jq -r 'unique | .[]' <<< "$numbers"); do
  pr=$(gh pr view "$number" --repo "$REPO" --json number,title,labels)
  confirmed=$(jq -c --argjson pending "$PENDING_LABELS" --argjson pr "$pr" '
    if ([$pr.labels[].name] | any(. as $l | $pending | index($l)))
    then . + [{number: $pr.number, title: $pr.title}]
    else . end
  ' <<< "$confirmed")
done
printf '%s\n' "$confirmed"
