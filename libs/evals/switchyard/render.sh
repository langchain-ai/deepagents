#!/usr/bin/env bash
# Render the six route configs from models.sh. Run after editing model ids.
#
#   ./render.sh
#
# Substitution is done by Python's string.Template rather than envsubst, which
# ships with gettext and is absent from a default macOS install. models.sh
# stays shell-sourced so its ${VAR:-default} forms still resolve.
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck source=/dev/null
set -a; . ./models.sh; set +a

if [ "$SY_WEAK_ID" = "FILL_ME_IN" ]; then
  echo "error: SY_WEAK_ID is still the placeholder. Set the Baseten model id in models.sh." >&2
  exit 1
fi

for arm in glm opus escalation nano glm-nano opus-nano; do
  python3 -c '
import os, string, sys
template = string.Template(sys.stdin.read())
try:
    sys.stdout.write(template.substitute(os.environ))
except KeyError as exc:
    sys.exit(f"error: models.sh does not define {exc.args[0]}")
' < "routes-${arm}.toml.tmpl" > "routes-${arm}.toml"
  echo "rendered routes-${arm}.toml"
done

echo
echo "Validate each config before spending a run on it:"
for arm in glm opus escalation nano glm-nano opus-nano; do
  echo "  switchyard-server --config routes-${arm}.toml --dry-run"
done
