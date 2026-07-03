#!/usr/bin/env bash
# Install deepagents-code.
#
# Usage:
#   curl -LsSf https://langch.in/dcode | bash
#
# Install an exact pre-release version:
#   curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_VERSION="0.1.0rc1" bash
#
# Override uv's pre-release strategy when resolving the latest version:
#   curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_PRERELEASE="allow" bash
#
# By default, the installer uses uv's `allow` pre-release strategy so stable
# deepagents-code releases that pin a pre-release dependency can resolve.
# DEEPAGENTS_CODE_VERSION and an explicit DEEPAGENTS_CODE_PRERELEASE are mutually
# exclusive: an exact pin already selects a single version, so setting both is an
# error.
#
# Already installed?
#   Safe to re-run. If a newer version exists, it asks before upgrading — or
#   upgrades on its own when run unattended (cron/CI/Docker). If you're already
#   on the latest, it does nothing. To skip the prompt:
#     - DEEPAGENTS_CODE_YES=1                     accept the upgrade
#     - DEEPAGENTS_CODE_VERSION / _PRERELEASE     install that exact selection
#     - DEEPAGENTS_CODE_EXTRAS / _PYTHON          rebuild with those options
#
# Uninstall:
#   This script installs deepagents-code as a uv tool. To remove it:
#     uv tool uninstall deepagents-code
#   That removes the dcode/deepagents-code binary and its isolated venv.
#   User config and data live separately in ~/.deepagents (config.toml,
#   hooks.json, a global .env, and a .state/ dir holding sessions and saved
#   credentials) and are NOT removed by the uninstall above. To also wipe them:
#     rm -rf ~/.deepagents
#   Optionally clear uv's shared tool cache (~/.cache/uv on Linux,
#   ~/Library/Caches/uv on macOS) — only if no other uv tools rely on it.
#
# Environment variables:
#   DEEPAGENTS_CODE_EXTRAS — comma-separated pip extras, e.g. "ollama",
#     "ollama,groq", or "daytona". Valid extras (see pyproject.toml for the
#     authoritative list):
#       Model providers: anthropic, baseten, bedrock, cohere, deepseek,
#         fireworks, google-genai, groq, huggingface, ibm, litellm, mistralai,
#         nvidia, ollama, openai, openrouter, perplexity, together, vertex, xai,
#         all-providers
#       Sandbox providers: agentcore, daytona, modal, runloop, vercel,
#         all-sandboxes
#       Standalone integrations: quickjs
#   DEEPAGENTS_CODE_VERSION — exact version to install, e.g. "0.1.0rc1"
#     (mutually exclusive with DEEPAGENTS_CODE_PRERELEASE)
#   DEEPAGENTS_CODE_PRERELEASE — uv pre-release strategy applied when
#     resolving the latest version: disallow, allow, if-necessary, explicit,
#     or if-necessary-or-explicit (default: allow; explicitly setting it is
#     mutually exclusive with DEEPAGENTS_CODE_VERSION)
#   DEEPAGENTS_CODE_PYTHON — Python version to use (default: 3.13)
#   DEEPAGENTS_CODE_YES — set to 1 to accept an available update without
#     prompting (assume "yes"). Exists so automated runs that still attach a
#     terminal (CI, wrapper scripts) update instead of stalling at the y/n
#     prompt.
#   DEEPAGENTS_CODE_SKIP_OPTIONAL — set to 1 to skip optional tool checks
#   DEEPAGENTS_CODE_RIPGREP_INSTALLER — how to provision ripgrep:
#     "managed" (default) eagerly installs the pinned, SHA-256-verified binary
#     into ~/.deepagents/bin (no sudo) via `dcode tools install`; "system"
#     keeps the interactive package-manager install (brew/apt/cargo/...). Set
#     DEEPAGENTS_CODE_OFFLINE=1 to skip the managed download entirely.
#   DEEPAGENTS_CODE_SKIP_XCODE_CHECK — set to 1 to bypass the macOS Xcode
#     Command Line Tools preflight check
#   DEEPAGENTS_CODE_VERBOSE — set to 1 to show uv's raw stderr (timing lines,
#     unfiltered package diff), the uv installer's own output (shown only when
#     uv isn't already installed), and the quiet-by-default status lines
#     (optional-tool checks, post-install footer); useful when debugging. A
#     fresh install otherwise hides the full list of installed dependencies.
#   UV_BIN — path to uv binary (auto-detected if unset)
#
# Credits:
#   Interactive mode detection, color logging, and optional tool install
#   patterns adapted from hermes-agent (NousResearch/hermes-agent).
#   Snap curl detection, shell-profile PATH modification, and symlink-first
#   PATH setup adapted from Amp (https://ampcode.com/install.sh).

set -euo pipefail

# Registry of temp files to clean up on exit or interrupt. Functions that
# create tempfiles append their paths here; cleanup_on_signal removes them all.
TEMP_FILES=()
register_temp() {
  TEMP_FILES+=("$1")
}
cleanup_temp_files() {
  for f in "${TEMP_FILES[@]:-}"; do
    rm -f "$f" 2>/dev/null || true
  done
}

# Keep the shell PATH the user started with. The installer may source
# ~/.local/bin/env later so it can find a freshly installed uv, but that does
# not update the parent shell that will receive the final "Run: dcode" advice.
ORIGINAL_PATH="${PATH:-}"

# ---------------------------------------------------------------------------
# Colors & logging
# ---------------------------------------------------------------------------
if [ -t 1 ] || [ "${FORCE_COLOR:-}" = "1" ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' CYAN='' BOLD='' NC=''
fi

log_info()    { printf "${CYAN}▸${NC} %s\n" "$*"; }
log_success() { printf "${GREEN}✔${NC} %s\n" "$*"; }
log_warn()    { printf "${YELLOW}⚠${NC} %s\n" "$*" >&2; }
log_error()   { printf "${RED}✖${NC} %s\n" "$*" >&2; }

# ---------------------------------------------------------------------------
# Exit / interrupt traps — ensures the user always sees an actionable message
# on failure and temp files are cleaned up on Ctrl-C / SIGTERM.
# ---------------------------------------------------------------------------
cleanup_on_signal() {
  local exit_code=$?
  cleanup_temp_files
  if [ $exit_code -ne 0 ]; then
    echo "" >&2
    log_error "Installation failed (exit code ${exit_code}). See errors above."
    log_error "For help, visit: https://docs.langchain.com/deepagents-code"
  fi
}
trap cleanup_on_signal EXIT

cleanup_on_interrupt() {
  # Disarm the EXIT trap first: exiting from here would otherwise also fire
  # cleanup_on_signal, appending a contradictory "Installation failed" message
  # after the friendly interrupt notice below. Temp files are still cleaned up
  # explicitly here, so nothing leaks despite the disarm.
  trap - EXIT
  echo "" >&2
  log_warn "Installation interrupted."
  cleanup_temp_files
  exit 1
}
trap cleanup_on_interrupt INT TERM

# ---------------------------------------------------------------------------
# Interactive mode detection
# ---------------------------------------------------------------------------
# When piped (curl | bash), stdin is not a terminal, but /dev/tty may still be
# available for prompts. IS_INTERACTIVE controls whether we ask the user
# questions; we never block a piped install on missing input.
IS_INTERACTIVE=false
if [ -t 0 ]; then
  IS_INTERACTIVE=true
elif [ -r /dev/tty ]; then
  # piped install but terminal is readable — can prompt via /dev/tty
  IS_INTERACTIVE=true
fi

# ---------------------------------------------------------------------------
# OS / platform detection
# ---------------------------------------------------------------------------
detect_os() {
  case "$(uname -s)" in
    Darwin)  OS="macos" ;;
    Linux)
             # shellcheck disable=SC2034
             # shellcheck disable=SC1091
             DISTRO=$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown}" || echo "unknown")
             OS="linux"
             ;;
    MINGW*|MSYS*|CYGWIN*)
             OS="windows" ;;
    *)       OS="unknown" ;;
  esac
}
detect_os

# ---------------------------------------------------------------------------
# macOS: require Xcode Command Line Tools
# ---------------------------------------------------------------------------
# On a fresh Mac the /usr/bin shims for git, python3, etc. are stubs that pop a
# blocking GUI dialog ("...requires the command line developer tools") the first
# time they run. uv's interpreter discovery and dcode's own git usage hit those
# stubs, so fail fast here with a clear instruction instead of leaving the user
# staring at a confusing popup mid-install. `xcode-select -p` only reports the
# active developer dir — it never triggers the install dialog itself.
if [ "$OS" = "macos" ] && [ "${DEEPAGENTS_CODE_SKIP_XCODE_CHECK:-}" != "1" ] && ! xcode-select -p >/dev/null 2>&1; then
  log_error "Xcode Command Line Tools are required but not installed."
  log_error "  Install them with:  xcode-select --install"
  log_error "  To bypass this check, set:  DEEPAGENTS_CODE_SKIP_XCODE_CHECK=1"
  log_error "  Then re-run this installer."
  exit 1
fi

# ---------------------------------------------------------------------------
# Root / MDM support (macOS — Kandji, Jamf, etc.)
# ---------------------------------------------------------------------------
# MDM tools run scripts as root in a minimal environment where HOME may be
# unset or point to /var/root.  Resolve the real console user's home so uv
# and dcode install to the right place.
if [ "$OS" = "macos" ] && { [ -z "${HOME:-}" ] || [ "$(id -u)" -eq 0 ]; }; then
  CONSOLE_USER="$(stat -f '%Su' /dev/console 2>/dev/null)" || {
    log_warn "Could not determine console user via /dev/console. Falling back to directory scan."
    CONSOLE_USER=""
  }

  if [ -n "$CONSOLE_USER" ] && [ "$CONSOLE_USER" != "root" ]; then
    if [ -d "/Users/$CONSOLE_USER" ]; then
      HOME="/Users/$CONSOLE_USER"
    else
      log_warn "Console user ${CONSOLE_USER} home /Users/${CONSOLE_USER} does not exist. Falling back to directory scan."
      CONSOLE_USER=""
    fi
  fi

  # Console user is root or undetectable (MDM enrollment, single-user mode,
  # headless session) — fall back to scanning /Users.
  if [ -z "${CONSOLE_USER:-}" ] || [ "$CONSOLE_USER" = "root" ]; then
    candidates="$(find /Users -mindepth 1 -maxdepth 1 -type d \
      ! -name root ! -name Shared ! -name '.*' | sort)"
    count="$(echo "$candidates" | grep -c . || true)"
    if [ "$count" -eq 1 ]; then
      HOME="$candidates"
    elif [ "$count" -gt 1 ]; then
      log_error "Multiple user directories found and no console user detected."
      log_error "  Set HOME explicitly: HOME=/Users/yourname curl ... | bash"
      exit 1
    else
      log_error "Could not determine user home directory. No user directories in /Users."
      exit 1
    fi
  fi

  export HOME
fi

# ---------------------------------------------------------------------------
# Ownership fix for root installs
# ---------------------------------------------------------------------------
# When running as root, files created under $HOME will be owned by root.
# Resolve the target user so we can fix ownership after install steps.
# When not root, fix_owner is a no-op.
if [ "$(id -u)" -eq 0 ]; then
  if [ "$OS" = "macos" ]; then
    # Reuse CONSOLE_USER from above; fall back to basename of the
    # already-resolved HOME (not a second stat call).
    TARGET_USER="${CONSOLE_USER:-$(basename "$HOME")}"
    [ "$TARGET_USER" = "root" ] && TARGET_USER="$(basename "$HOME")"
  else
    TARGET_USER="${SUDO_USER:-$(basename "$HOME")}"
  fi

  if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
    log_warn "Could not determine non-root target user. Files under ${HOME} may remain owned by root."
    log_warn "  After install, run: sudo chown -R YOUR_USERNAME ~/.local"
    fix_owner() { :; }
  else
    fix_owner() {
      if ! chown -R "$TARGET_USER" "$@" 2>&1; then
        log_warn "Could not fix ownership of $* for user ${TARGET_USER}."
      fi
    }
  fi
else
  fix_owner() { :; }
fi

# ---------------------------------------------------------------------------
# Prompt helper — reads from /dev/tty when stdin is piped
# ---------------------------------------------------------------------------
prompt_yn() {
  local question="$1"
  if [ "$IS_INTERACTIVE" = false ]; then
    return 1
  fi
  local reply
  if [ -t 0 ]; then
    printf "%s [y/N] " "$question"
    read -r reply
  else
    printf "%s [y/N] " "$question" > /dev/tty
    if ! read -r reply < /dev/tty 2>/dev/null; then
      log_warn "Could not read from /dev/tty — skipping prompt."
      return 1
    fi
  fi
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    return 0
  fi
  return 1
}

# Whether an interactive y/n prompt can actually be answered. IS_INTERACTIVE
# trusts `[ -r /dev/tty ]`, which only access-checks the device — opening it
# still fails when there is no controlling terminal (cron, systemd, some CI).
# Confirm the channel is usable so callers can fall back instead of blocking
# or silently treating an unanswerable prompt as "no".
can_prompt() {
  [ "$IS_INTERACTIVE" = true ] || return 1
  [ -t 0 ] && return 0
  { : < /dev/tty; } 2>/dev/null
}

path_is_under_home() {
  local path="$1"
  local home_real=""
  local path_real=""
  [ -n "${HOME:-}" ] || return 1
  [ -d "$path" ] || return 1
  home_real=$(cd "$HOME" 2>/dev/null && pwd -P) || return 1
  path_real=$(cd "$path" 2>/dev/null && pwd -P) || return 1
  case "$path_real" in
    "$home_real"/*) return 0 ;;
    *) return 1 ;;
  esac
}

prepare_install_log_dir() {
  local cache_root="$1"
  local dir="${cache_root}/deepagents-code"
  [ -n "$cache_root" ] || return 1
  [ ! -L "$cache_root" ] || return 1
  [ ! -L "$dir" ] || return 1
  if [ ! -d "$cache_root" ]; then
    mkdir -m 700 -p "$cache_root" 2>/dev/null || return 1
  fi
  [ -d "$cache_root" ] && [ ! -L "$cache_root" ] || return 1
  if [ -e "$dir" ]; then
    [ -d "$dir" ] && [ ! -L "$dir" ] || return 1
  else
    mkdir -m 700 "$dir" 2>/dev/null || return 1
  fi
  if [ "$(id -u)" -eq 0 ]; then
    path_is_under_home "$dir" || return 1
  fi
  printf '%s\n' "$dir"
}

fix_install_log_owner() {
  [ -n "${INSTALL_LOG:-}" ] || return 0
  [ "$(id -u)" -eq 0 ] || return 0
  [ -n "${TARGET_USER:-}" ] && [ "$TARGET_USER" != "root" ] || return 0
  [ -d "$install_log_dir" ] && [ ! -L "$install_log_dir" ] || return 0
  path_is_under_home "$install_log_dir" || return 0
  if ! chown -h "$TARGET_USER" "$install_log_dir" 2>&1; then
    log_warn "Could not fix ownership of $install_log_dir for user ${TARGET_USER}."
  fi
  if [ -f "$INSTALL_LOG" ] && [ ! -L "$INSTALL_LOG" ]; then
    if ! chown -h "$TARGET_USER" "$INSTALL_LOG" 2>&1; then
      log_warn "Could not fix ownership of $INSTALL_LOG for user ${TARGET_USER}."
    fi
  fi
}

copy_install_log() {
  [ -n "${INSTALL_LOG:-}" ] || return 1
  [ -n "${install_log_dir:-}" ] || return 1
  [ -d "$install_log_dir" ] && [ ! -L "$install_log_dir" ] || return 1
  if [ "$(id -u)" -eq 0 ]; then
    path_is_under_home "$install_log_dir" || return 1
  fi
  [ ! -L "$INSTALL_LOG" ] || return 1
  rm -f "$INSTALL_LOG" 2>/dev/null || return 1
  # Publish the already-captured stderr without opening the destination for
  # writing. `ln` fails if an attacker wins the race by creating install.log.
  ln "$uv_stderr" "$INSTALL_LOG" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXTRAS="${DEEPAGENTS_CODE_EXTRAS:-}"
VERSION="${DEEPAGENTS_CODE_VERSION:-}"
PRERELEASE_REQUESTED="${DEEPAGENTS_CODE_PRERELEASE:-}"
PRERELEASE="${PRERELEASE_REQUESTED:-allow}"
PYTHON_REQUESTED=false
if [[ -n "${DEEPAGENTS_CODE_PYTHON:-}" ]]; then
  PYTHON_REQUESTED=true
fi
PYTHON_VERSION="${DEEPAGENTS_CODE_PYTHON:-3.13}"
SKIP_OPTIONAL="${DEEPAGENTS_CODE_SKIP_OPTIONAL:-0}"
VERBOSE="${DEEPAGENTS_CODE_VERBOSE:-0}"
ASSUME_YES="${DEEPAGENTS_CODE_YES:-0}"
# How ripgrep gets provisioned: "managed" (default) eagerly fetches the
# pinned, SHA-256-verified binary into ~/.deepagents/bin via `dcode tools
# install`; "system" keeps the interactive package-manager path below. Any
# value other than "system" normalizes to "managed".
#
# Lowercase and strip whitespace first so this matches the `.strip().lower()`
# normalization in managed_tools.ripgrep_installer(). Without this, a value
# like "System" would parse as "managed" here but "system" in dcode, and the
# eager `dcode tools install` would skip silently while this script also
# skipped the package-manager path — leaving ripgrep unprovisioned.
RIPGREP_INSTALLER="$(printf '%s' "${DEEPAGENTS_CODE_RIPGREP_INSTALLER:-managed}" \
  | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
case "$RIPGREP_INSTALLER" in
  system) RIPGREP_INSTALLER="system" ;;
  *)      RIPGREP_INSTALLER="managed" ;;
esac

# PyPI JSON endpoint used to discover the latest published release so we can
# tell whether an existing install is out of date before upgrading it.
PYPI_JSON_URL="https://pypi.org/pypi/deepagents-code/json"

# Validate and normalize extras: accept bare CSV, wrap in brackets for pip
if [[ -n "$EXTRAS" ]]; then
  # Strip brackets if the user passed them anyway
  EXTRAS="${EXTRAS#[}"
  EXTRAS="${EXTRAS%]}"
  if [[ ! "$EXTRAS" =~ ^[-a-zA-Z0-9,]+$ ]]; then
    log_error "DEEPAGENTS_CODE_EXTRAS must be comma-separated extra names, e.g. 'anthropic,groq' or 'daytona'"
    exit 1
  fi
  EXTRAS="[${EXTRAS}]"
fi

# An exact pin already selects a single version, so an explicitly requested
# pre-release strategy (which only affects how a range resolves) is redundant at
# best and contradictory at worst (e.g. an rc pin with "disallow"). Reject only
# user-provided combinations; the installer's default `if-necessary` strategy is
# not forwarded when a version is pinned.
if [[ -n "$VERSION" && -n "$PRERELEASE_REQUESTED" ]]; then
  log_error "DEEPAGENTS_CODE_VERSION and DEEPAGENTS_CODE_PRERELEASE are mutually exclusive."
  log_error "Pin an exact version, or set a pre-release strategy — not both."
  exit 1
fi

VERSION_SPEC=""
if [[ -n "$VERSION" ]]; then
  # Require a leading alphanumeric so the value reads as a version rather than
  # an option (e.g. "-U"); the class excludes every shell metacharacter, so the
  # value is safe to interpolate into the single argv token passed to uv.
  if [[ ! "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9_.!+-]*$ ]]; then
    log_error "DEEPAGENTS_CODE_VERSION must be an exact version, e.g. '0.1.0rc1'"
    exit 1
  fi
  VERSION_SPEC="==${VERSION}"
fi

if [[ -n "$PRERELEASE" ]]; then
  case "$PRERELEASE" in
    disallow|allow|if-necessary|explicit|if-necessary-or-explicit)
      ;;
    *)
      log_error "Invalid DEEPAGENTS_CODE_PRERELEASE."
      log_error "Use: disallow, allow, if-necessary, explicit, or if-necessary-or-explicit"
      exit 1
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# uv installation
# ---------------------------------------------------------------------------

# Detect whether `curl` is a snap package, which lacks the permissions to
# download files outside the snap sandbox. On such systems curl appears to
# work but fails on actual downloads, so callers should fall back to wget.
is_snap_curl() {
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  local curl_path
  curl_path=$(command -v curl 2>/dev/null) || return 1
  case "$curl_path" in
    */snap/*) return 0 ;;
    *)        return 1 ;;
  esac
}

# Download a URL to stdout using the first available working downloader.
# Prefers curl (unless it's a snap install, which has sandbox permission
# issues), then falls back to wget. Prints nothing and returns non-zero if no
# working downloader is available.
download_to_stdout() {
  local url="$1" ua="${2:-deepagents-code-install}"
  if command -v curl >/dev/null 2>&1 && ! is_snap_curl; then
    curl -fsSL -H "User-Agent: ${ua}" "$url" 2>/dev/null || return $?
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- --header="User-Agent: ${ua}" "$url" 2>/dev/null || return $?
  else
    return 1
  fi
}

install_uv() {
  # The upstream uv installer is chatty (download progress, install paths,
  # PATH-setup hints). Capture it and surface the output only when debugging
  # or when the install fails — by default it's noise the user doesn't need.
  local uv_install_out uv_install_rc=0
  uv_install_out=$(mktemp 2>/dev/null) || {
    log_error "mktemp is required to create a secure temp file."
    exit 1
  }
  register_temp "$uv_install_out"
  # Only the piped `sh` (the installer body) is captured by `>"$uv_install_out"
  # 2>&1`; curl/wget keep their own stderr on the terminal. That's intentional —
  # `-fsSL` includes `-S`, so a failed download still prints curl's error
  # directly (above the "uv installation failed" line) even though the captured
  # file is then empty. Don't assume curl's stderr is in the capture.
  if command -v curl >/dev/null 2>&1 && ! is_snap_curl; then
    curl -fsSL https://astral.sh/uv/install.sh | sh >"$uv_install_out" 2>&1 || uv_install_rc=$?
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh >"$uv_install_out" 2>&1 || uv_install_rc=$?
  elif is_snap_curl; then
    rm -f "$uv_install_out"
    log_error "curl is installed as a snap and cannot download files due to sandbox permissions."
    log_error "Please install wget, or reinstall curl with a different package manager (e.g. apt)."
    exit 1
  else
    rm -f "$uv_install_out"
    log_error "curl or wget is required to install uv."
    exit 1
  fi
  if [ "$VERBOSE" = "1" ] || [ "$uv_install_rc" -ne 0 ]; then
    cat "$uv_install_out" >&2
  fi
  rm -f "$uv_install_out"
  if [ "$uv_install_rc" -ne 0 ]; then
    log_error "uv installation failed. See errors above."
    exit 1
  fi
}

# Resolve uv binary: honor UV_BIN override, then PATH, the env file written by
# uv's installer, then the default install location (~/.local/bin). MDM and cron
# jobs often run with a minimal PATH, so an existing uv in ~/.local/bin must
# count as installed before we invoke the upstream installer.
resolve_uv_bin() {
  if [ -n "${UV_BIN:-}" ]; then
    case "$UV_BIN" in
      */*) [ -f "$UV_BIN" ] && [ -x "$UV_BIN" ] ;;
      *)   command -v "$UV_BIN" >/dev/null 2>&1 ;;
    esac
    return $?
  fi

  if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
    return 0
  fi

  if [ -f "${HOME}/.local/bin/env" ]; then
    set +e +u
    # shellcheck source=/dev/null
    . "${HOME}/.local/bin/env"
    set -e -u
    if command -v uv >/dev/null 2>&1; then
      UV_BIN="uv"
      return 0
    fi
  fi

  if [ -x "${HOME}/.local/bin/uv" ]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi

  return 1
}

if ! resolve_uv_bin; then
  if [ -n "${UV_BIN:-}" ]; then
    log_error "UV_BIN is set but does not point to an executable uv: ${UV_BIN}"
    exit 1
  fi
  log_info "uv not found — installing..."
  install_uv
  fix_owner "${HOME}/.local/bin"  # root installs: restore user ownership
  if ! resolve_uv_bin; then
    log_error "uv not found after installation. Restart your shell or add ~/.local/bin to PATH."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Latest-version lookup
# ---------------------------------------------------------------------------
# Print the latest published deepagents-code version from PyPI, or nothing on
# any failure (offline, transient error, missing downloader). PyPI nests the
# latest release at "info.version"; that key appears first in the response (the
# "info" object leads), so taking the first "version" match selects it without
# depending on a JSON parser. The pattern tolerates whitespace around the colon
# so a switch to pretty-printed JSON wouldn't silently break the probe.
# This relies on PyPI's current (not contractually guaranteed) key ordering; if
# it ever changed, the worst case is a wrong/empty match, which the caller
# already treats as "unknown latest" and recovers from — never a bad install.
fetch_latest_version() {
  local json="" ua="deepagents-code-install"
  json=$(download_to_stdout "$PYPI_JSON_URL" "$ua" 2>/dev/null) || return 0
  if [ -z "$json" ]; then
    return 0
  fi
  # `|| true` keeps a no-match (grep exit 1 under `pipefail`) from aborting the
  # script; an empty result is handled by the caller as "unknown latest".
  printf '%s' "$json" \
    | grep -oE '"version"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | head -1 \
    | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/' || true
}

# ---------------------------------------------------------------------------
# Install deepagents-code
# ---------------------------------------------------------------------------
PACKAGE="deepagents-code${EXTRAS}${VERSION_SPEC}"

# Capture pre-install version (if any) for messaging
PRE_VERSION=""
for candidate in dcode deepagents-code; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PRE_VERSION=$("$candidate" -v 2>/dev/null | head -1 | awk '{print $NF}') || PRE_VERSION=""
    break
  elif [ -x "${HOME}/.local/bin/${candidate}" ]; then
    PRE_VERSION=$("${HOME}/.local/bin/${candidate}" -v 2>/dev/null | head -1 | awk '{print $NF}') || PRE_VERSION=""
    break
  fi
done

# Detect editable installs (uv tool install -e <path>) so we can tell the user
# why the environment will be rebuilt instead of upgraded in place.
IS_EDITABLE=false
EDITABLE_SRC=""
UV_TOOL_DIR=""
if UV_TOOL_DIR_RAW=$("$UV_BIN" tool dir 2>/dev/null); then
  UV_TOOL_DIR="$UV_TOOL_DIR_RAW"
fi
if [ -n "$UV_TOOL_DIR" ] && [ -d "${UV_TOOL_DIR}/deepagents-code" ]; then
  shopt -s nullglob
  for du in "${UV_TOOL_DIR}"/deepagents-code/lib/python*/site-packages/deepagents_code-*.dist-info/direct_url.json; do
    if grep -q '"editable"[[:space:]]*:[[:space:]]*true' "$du" 2>/dev/null; then
      IS_EDITABLE=true
      EDITABLE_SRC=$(sed -nE 's|.*"url"[[:space:]]*:[[:space:]]*"file://([^"]*)".*|\1|p' "$du" | head -1)
      # Guard against malformed JSON producing a bogus path.
      [ -n "$EDITABLE_SRC" ] && [ ! -d "$EDITABLE_SRC" ] && EDITABLE_SRC=""
      break
    fi
  done
  shopt -u nullglob
fi

if [ "$IS_EDITABLE" = true ]; then
  pre_label="${PRE_VERSION:-(version unknown)}"
  if [ -n "$EDITABLE_SRC" ]; then
    log_info "deepagents-code ${pre_label} found (editable install from ${EDITABLE_SRC})."
  else
    log_info "deepagents-code ${pre_label} found (editable install from local source)."
  fi
  log_info "  Replacing with a standard install from PyPI — the existing environment will be rebuilt."
elif [ -n "$PRE_VERSION" ] && [ -z "$VERSION" ] && [ -z "$PRERELEASE_REQUESTED" ]; then
  # Default path with an existing install: probe PyPI and prompt before
  # upgrading, rather than silently pulling the latest version every run.
  # A pinned version or pre-release strategy (handled by the branches above and
  # below) expresses explicit intent, so those install directly.
  #
  # The up-to-date check below is plain string equality, so it relies on
  # PRE_VERSION (the raw `dcode -v` literal) and LATEST_VERSION (PyPI's
  # PEP 440-normalized `info.version`) being identically canonical. release-please
  # keeps `_version.py` to clean `X.Y.Z`, so they match today; a non-canonical
  # release literal would merely re-prompt an up-to-date user, never silently
  # skip a real upgrade. A shell installer can't import `packaging` to compare
  # semantically the way `update_check.py` does.
  log_info "deepagents-code ${PRE_VERSION} found — checking for updates..."
  LATEST_VERSION=$(fetch_latest_version)
  if [ -z "$LATEST_VERSION" ]; then
    log_warn "Could not determine the latest version from PyPI — continuing with an upgrade attempt."
  elif [ -n "$EXTRAS" ] || [ "$PYTHON_REQUESTED" = true ]; then
    if [ "$LATEST_VERSION" = "$PRE_VERSION" ]; then
      log_info "deepagents-code is already up to date — rebuilding with requested options."
    else
      log_info "Updating deepagents-code ${PRE_VERSION} → ${LATEST_VERSION} with requested options..."
    fi
  elif [ "$LATEST_VERSION" = "$PRE_VERSION" ]; then
    log_success "deepagents-code is already up to date."
    exit 0
  elif [ "$ASSUME_YES" = "1" ]; then
    log_info "Updating deepagents-code ${PRE_VERSION} → ${LATEST_VERSION}..."
  elif can_prompt; then
    if prompt_yn "Update deepagents-code ${PRE_VERSION} → ${LATEST_VERSION}?"; then
      log_info "Updating deepagents-code ${PRE_VERSION} → ${LATEST_VERSION}..."
    else
      log_info "Keeping deepagents-code ${PRE_VERSION}. Re-run this installer anytime to update."
      exit 0
    fi
  else
    # No TTY to prompt (cron, CI, Dockerfile RUN, systemd): there is no human to
    # ask, and an installer's job is to make the current version present, so
    # complete the upgrade rather than silently no-op. Callers that want a fixed
    # version pin DEEPAGENTS_CODE_VERSION, which skips this path entirely.
    log_info "deepagents-code ${LATEST_VERSION} available — updating (no TTY to prompt)."
  fi
elif [ -n "$PRE_VERSION" ]; then
  log_info "deepagents-code ${PRE_VERSION} found — checking for updates..."
else
  log_info "Installing ${PACKAGE}..."
fi

# Capture uv stderr so we can:
#   1. Rewrite the cryptic "Ignoring existing environment ..." warning into
#      plain English. uv emits that line when it rebuilds the tool venv
#      instead of upgrading in place (e.g., Python interpreter mismatch, or
#      editable↔regular install swap).
#   2. Drop uv's per-step timing lines ("Resolved N packages in...", etc.)
#      download/build progress, and the trailing "Installed N executables:" line
#      — we already show a concise install/update summary.
#   3. Reformat the `- pkg==X` / `+ pkg==Y` diff into an aligned
#      "pkg  X → Y" table under a single header.
#   4. Detect whether uv actually moved any packages (those same
#      `- pkg==X` / `+ pkg==Y` lines). A same-version reinstall that still
#      bumps dependencies must report differently from a true no-op, so a
#      later grep over this raw tempfile sets UV_REPORTED_PACKAGE_CHANGES.
#   5. Persist the raw output to a log file (see INSTALL_LOG below) so a
#      same-version dependency bump — or a failed install — can point the
#      user at the full details after the terminal scrolls away.
# Using a tempfile (vs. process substitution) ensures we see uv's full exit
# status, don't race the warning past later log lines, and can re-scan the
# raw output for (4) after the awk pass above has already reformatted it.
uv_stderr=$(mktemp 2>/dev/null) || uv_stderr="/tmp/deepagents-install.$$.err"
register_temp "$uv_stderr"
uv_rc=0
UV_REPORTED_PACKAGE_CHANGES=false
# Mirror uv's raw output to a persistent log under the XDG cache dir. A
# same-version dependency bump prints only a one-line summary and a failed
# install scrolls past, so the log preserves the full diff/errors for later.
# Prefer $XDG_CACHE_HOME, falling back to ~/.cache. INSTALL_LOG is the real
# path used for writes; INSTALL_LOG_DISPLAY is the tilde-collapsed form shown
# to the user. Both stay empty when the dir can't be created, which every
# consumer treats as "feature disabled" so messages degrade cleanly.
INSTALL_LOG=""
INSTALL_LOG_DISPLAY=""
cache_root="${XDG_CACHE_HOME:-}"
if [ "$(id -u)" -eq 0 ] && [ -n "${HOME:-}" ]; then
  cache_root="${HOME}/.cache"
elif [ -z "$cache_root" ] && [ -n "${HOME:-}" ]; then
  cache_root="${HOME}/.cache"
fi
if [ -n "$cache_root" ]; then
  if install_log_dir=$(prepare_install_log_dir "$cache_root"); then
    INSTALL_LOG="${install_log_dir}/install.log"
    INSTALL_LOG_DISPLAY="$INSTALL_LOG"
    if [ -n "${HOME:-}" ]; then
      case "$INSTALL_LOG" in
        "$HOME"/*) INSTALL_LOG_DISPLAY="~${INSTALL_LOG#"$HOME"}" ;;
      esac
    fi
  fi
fi
if [[ -z "$VERSION" ]]; then
  "$UV_BIN" tool install -U --python "$PYTHON_VERSION" \
    --prerelease "$PRERELEASE" "$PACKAGE" 2>"$uv_stderr" || uv_rc=$?
else
  "$UV_BIN" tool install -U --python "$PYTHON_VERSION" "$PACKAGE" \
    2>"$uv_stderr" || uv_rc=$?
fi
if [ "$VERBOSE" != "1" ] && command -v awk >/dev/null 2>&1; then
  awk '
    /^Ignoring existing environment/ {
      print "⚠ Existing environment uses a different Python — rebuilding from scratch (this is normal)."
      next
    }
    /^Resolved( [0-9]+ packages?)? in /     { next }
    /^Prepared [0-9]+ packages?( |$)/       { next }
    /^Uninstalled [0-9]+ packages? in /     { next }
    /^Installed [0-9]+ packages? in /       { next }
    /^Audited( [0-9]+ packages?)? in /      { next }
    /^Checked( [0-9]+ packages?)? in /      { next }
    /^[[:space:]]*Downloading /         { next }
    /^[[:space:]]*Downloaded /          { next }
    /^[[:space:]]*Building /            { next }
    /^[[:space:]]*Built /                { next }
    /^Installed [0-9]+ executables?:/   { next }
    /^ - / {
      s = $0; sub(/^ - /, "", s); n = index(s, "==")
      if (n > 0) {
        pkg = substr(s, 1, n - 1); ver = substr(s, n + 2)
        removed[pkg] = ver
        if (!(pkg in seen)) { seen[pkg] = 1; order[++cnt] = pkg }
      }
      next
    }
    /^ \+ / {
      s = $0; sub(/^ \+ /, "", s); n = index(s, "==")
      if (n > 0) {
        pkg = substr(s, 1, n - 1); ver = substr(s, n + 2)
        added[pkg] = ver
        if (!(pkg in seen)) { seen[pkg] = 1; order[++cnt] = pkg }
      }
      next
    }
    { print }
    END {
      if (cnt == 0) exit
      any_removed = 0
      for (i = 1; i <= cnt; i++) {
        if (order[i] in removed) any_removed = 1
      }
      if (!any_removed) {
        # No upgrades or removals — every touched package is a brand-new
        # addition (a fresh install, or new extras pulled into an existing
        # env). Listing the full transitive set is noise; verbose mode keeps
        # the output available for debugging.
        exit
      }
      maxw = 0
      for (i = 1; i <= cnt; i++) {
        p = order[i]
        if (length(p) > maxw) maxw = length(p)
      }
      # Upgrades touch only a handful of packages, so the diff stays compact and
      # genuinely useful — keep printing it. "(new)" disambiguates added rows
      # from upgraded/removed ones within this mixed list.
      print "Updated packages:"
      for (i = 1; i <= cnt; i++) {
        p = order[i]
        pad = ""
        for (j = length(p); j < maxw; j++) pad = pad " "
        if ((p in removed) && (p in added)) {
          printf "  %s%s  %s → %s\n", p, pad, removed[p], added[p]
        } else if (p in added) {
          printf "  %s%s  %s (new)\n", p, pad, added[p]
        } else {
          printf "  %s%s  %s (removed)\n", p, pad, removed[p]
        }
      }
    }
  ' "$uv_stderr" >&2
else
  cat "$uv_stderr" >&2
fi
if grep -Eq '^[[:space:]]+[-+][[:space:]]+[^=]+==' "$uv_stderr"; then
  UV_REPORTED_PACKAGE_CHANGES=true
fi
if [ -n "$INSTALL_LOG" ]; then
  copy_install_log || { INSTALL_LOG=""; INSTALL_LOG_DISPLAY=""; }
fi
rm -f "$uv_stderr"
if [ "$uv_rc" -ne 0 ]; then
  log_error "Failed to install ${PACKAGE}. See errors above."
  # The log captured uv's full stderr (copied just above, before this exit), so
  # point the user at it — non-verbose mode trims uv's lines from the terminal
  # and piped `curl | bash` runs lose scrollback.
  if [ -n "$INSTALL_LOG" ]; then
    log_error "Full install log: ${INSTALL_LOG_DISPLAY}"
  fi
  log_error "Common fixes: check your network, try a different Python version (DEEPAGENTS_CODE_PYTHON=3.12), or install manually."
  exit 1
fi
fix_owner "${HOME}/.local/bin" "${HOME}/.local/share/uv"  # uv binaries + tool data
if [ "$OS" = "macos" ] && [ -d "${HOME}/Library/Caches/uv" ]; then
  fix_owner "${HOME}/Library/Caches/uv"
elif [ -d "${HOME}/.cache/uv" ]; then
  fix_owner "${HOME}/.cache/uv"
fi
# Restore ownership for the log path without recursively chowning a cache path
# that could have been swapped after creation.
fix_install_log_owner

# ---------------------------------------------------------------------------
# PATH setup — make dcode immediately findable in a new shell
# ---------------------------------------------------------------------------
# After `uv tool install`, dcode lands in ~/.local/bin. If that directory is
# already in the user's PATH (via ~/.local/bin/env or a shell profile), dcode
# just works after a shell restart. If it isn't, the user is stuck with a
# successful install but no callable binary.
#
# Strategy (adapted from Amp's installer, https://ampcode.com/install.sh):
#   1. If a common bin dir (~/.local/bin, ~/bin, ~/.bin) is already in PATH,
#      create a symlink there — no profile modification needed.
#   2. Otherwise, create ~/.local/bin, symlink dcode there, then add
#      ~/.local/bin to the user's shell profile (.zshrc, .bashrc,
#      .bash_profile, or config.fish). Prompt interactively before writing;
#      auto-add in non-interactive mode (CI, cron, piped install).
#   3. Skip the whole thing if the binary is already on PATH or uv's env file
#      exists (uv's installer already handles PATH setup in that case).

# Check if a directory is in PATH.
dir_in_path() {
  local check_dir="$1"
  [ -d "$check_dir" ] || return 1
  check_dir=$(cd "$check_dir" 2>/dev/null && pwd) || return 1
  case ":${PATH:-}:" in
    *":$check_dir:"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Try to symlink the dcode binary into a directory already in PATH. Tries
# ~/.local/bin, ~/bin, and ~/.bin in order. Returns 0 on success.
try_symlink_in_path() {
  local binary_name="$1"
  local binary_path="$2"
  local preferred_dirs=("$HOME/.local/bin" "$HOME/bin" "$HOME/.bin")
  local dir symlink_path
  for dir in "${preferred_dirs[@]}"; do
    if dir_in_path "$dir"; then
      mkdir -p "$dir" 2>/dev/null || continue
      symlink_path="$dir/$binary_name"
      if [ "$binary_path" = "$symlink_path" ]; then
        return 0
      fi
      # Remove existing symlink if it points elsewhere or is stale
      if [ -L "$symlink_path" ]; then
        rm -f "$symlink_path"
      fi
      if ln -sf "$binary_path" "$symlink_path" 2>/dev/null; then
        fix_owner "$symlink_path" 2>/dev/null || true
        return 0
      fi
    fi
  done
  return 1
}

# Detect the user's shell and return the profile file + PATH export statement.
# Sets SHELL_PROFILE and PATH_EXPORT as globals.
detect_shell_profile() {
  local default_shell="bash"
  if [ "$OS" = "macos" ]; then
    default_shell="zsh"
  fi
  local shell_name
  shell_name=$(basename "${SHELL:-$default_shell}" 2>/dev/null) || shell_name="$default_shell"
  SHELL_PROFILE=""
  PATH_EXPORT=""
  case "$shell_name" in
    zsh)
      SHELL_PROFILE="$HOME/.zshrc"
      # shellcheck disable=SC2016  # single-quoted so $HOME/$PATH expand at profile source time, not here
      PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'
      ;;
    bash)
      if [ "$OS" = "macos" ]; then
        if [ -f "$HOME/.bash_profile" ]; then
          SHELL_PROFILE="$HOME/.bash_profile"
        elif [ -f "$HOME/.bashrc" ]; then
          SHELL_PROFILE="$HOME/.bashrc"
        else
          SHELL_PROFILE="$HOME/.bash_profile"
        fi
      else
        if [ -f "$HOME/.bashrc" ]; then
          SHELL_PROFILE="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
          SHELL_PROFILE="$HOME/.bash_profile"
        else
          SHELL_PROFILE="$HOME/.bashrc"
        fi
      fi
      # shellcheck disable=SC2016  # single-quoted so $HOME/$PATH expand at profile source time, not here
      PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'
      ;;
    fish)
      SHELL_PROFILE="$HOME/.config/fish/config.fish"
      # shellcheck disable=SC2016  # single-quoted so $HOME expands at profile source time, not here
      PATH_EXPORT='fish_add_path "$HOME/.local/bin"'
      ;;
    *)
      # Unknown shell — don't modify any profile.
      ;;
  esac
}

# Check if ~/.local/bin is already referenced in the shell profile's PATH
# config. Matches non-commented lines containing .local/bin in a PATH
# assignment or fish_add_path. Returns 0 if already present.
local_bin_in_profile() {
  local profile="$1"
  [ -f "$profile" ] || return 1
  grep -v '^[[:space:]]*#' "$profile" 2>/dev/null | grep -qE 'PATH=.*\.local/bin' \
    || grep -v '^[[:space:]]*#' "$profile" 2>/dev/null | grep -qE 'fish_add_path.*\.local/bin'
}

# Ensure dcode is on PATH for new shell sessions. Creates symlinks and/or
# modifies the shell profile as needed. Only acts when the binary verified
# but isn't already on the user's original PATH.
ensure_path_setup() {
  local binary_name="$1"
  local binary_path="$2"

  # uv's env file already handles PATH setup — no need to duplicate.
  if [ -f "$HOME/.local/bin/env" ]; then
    return 0
  fi

  # Step 1: try symlinking into a dir already in PATH (no profile change).
  if try_symlink_in_path "$binary_name" "$binary_path"; then
    if [ "$VERBOSE" = "1" ]; then
      log_success "Created symlink in PATH for ${binary_name}."
    fi
    return 0
  fi

  # Step 2: create ~/.local/bin, symlink there, then add to shell profile.
  mkdir -p "$HOME/.local/bin" 2>/dev/null || {
    log_warn "Could not create ~/.local/bin."
    return 1
  }
  fix_owner "$HOME/.local/bin"
  local symlink_path="$HOME/.local/bin/$binary_name"
  if [ "$binary_path" != "$symlink_path" ]; then
    if [ -L "$symlink_path" ]; then
      rm -f "$symlink_path"
    fi
    if ! ln -sf "$binary_path" "$symlink_path" 2>/dev/null; then
      log_warn "Could not create symlink at ${symlink_path}."
      return 1
    fi
    fix_owner "$symlink_path"
  fi

  # Step 3: detect shell and add ~/.local/bin to profile if needed.
  detect_shell_profile
  if [ -z "$SHELL_PROFILE" ]; then
    log_warn "${binary_name} installed to ~/.local/bin but your shell is unknown."
    log_warn "  Add ~/.local/bin to your PATH manually."
    return 1
  fi

  # Already in profile? Just tell the user to reload.
  if local_bin_in_profile "$SHELL_PROFILE"; then
    if [ "$VERBOSE" = "1" ]; then
      # shellcheck disable=SC2088  # display string, literal ~/ is intended for readability
      log_info "~/.local/bin already in ${SHELL_PROFILE}."
    fi
    return 0
  fi

  # Collapse $HOME prefix to ~ for a tidier display path.
  local tilde_profile="${SHELL_PROFILE/#$HOME/\~}"

  # Prompt interactively, or auto-add when non-interactive.
  local should_add=true
  if [ "$IS_INTERACTIVE" = true ] && can_prompt; then
    if ! prompt_yn "Add ~/.local/bin to your PATH in ${tilde_profile}?"; then
      should_add=false
    fi
  fi

  if [ "$should_add" = true ]; then
    # Create the profile file if it doesn't exist.
    if [ ! -f "$SHELL_PROFILE" ]; then
      mkdir -p "$(dirname "$SHELL_PROFILE")" 2>/dev/null || true
      touch "$SHELL_PROFILE" 2>/dev/null || {
        log_warn "Could not create ${tilde_profile}. Add ~/.local/bin to PATH manually."
        return 1
      }
    fi
    {
      echo ""
      echo "# Added by deepagents-code installer"
      echo "$PATH_EXPORT"
    } >> "$SHELL_PROFILE"
    fix_owner "$SHELL_PROFILE"
    log_success "Added ~/.local/bin to PATH in ${tilde_profile}."
  else
    log_info "Skipped modifying ${tilde_profile}."
    log_info "  To use ${binary_name}, add to PATH:  ${PATH_EXPORT}"
  fi
}
DCODE_BIN=""
DCODE_NAME=""
# Tracks whether the binary would have resolved via the user's original PATH,
# not the installer-mutated PATH. A fresh `uv tool install` drops the binary in
# ~/.local/bin, and this script may have sourced ~/.local/bin/env earlier to
# find uv; the parent shell still won't have dcode on PATH until it is
# restarted or the env file is sourced.
DCODE_ON_PATH=false
for candidate in dcode deepagents-code; do
  if resolved=$(command -v "$candidate" 2>/dev/null) && [ -n "$resolved" ]; then
    DCODE_BIN="$resolved"
    DCODE_NAME="$candidate"
    if PATH="$ORIGINAL_PATH" command -v "$candidate" >/dev/null 2>&1; then
      DCODE_ON_PATH=true
    fi
    break
  elif [ -x "${HOME}/.local/bin/${candidate}" ]; then
    DCODE_BIN="${HOME}/.local/bin/${candidate}"
    DCODE_NAME="$candidate"
    break
  fi
done

# Collapse $HOME prefix to ~ for a tidier display path. Used in user-facing
# log lines only; DCODE_BIN keeps the absolute path for any exec needs.
DCODE_BIN_DISPLAY="$DCODE_BIN"
if [ -n "$DCODE_BIN" ] && [ -n "${HOME:-}" ]; then
  case "$DCODE_BIN" in
    "$HOME"/*) DCODE_BIN_DISPLAY="~${DCODE_BIN#"$HOME"}" ;;
  esac
fi

NEW_VERSION=""
VERIFY_OK=false
VERIFY_OUTPUT=""
if [ -n "$DCODE_BIN" ]; then
  if VERIFY_OUTPUT=$("$DCODE_BIN" -v 2>&1); then
    NEW_VERSION=$(printf '%s\n' "$VERIFY_OUTPUT" | head -1 | awk '{print $NF}') || NEW_VERSION=""
    VERIFY_OK=true
  fi
fi

if [ "$IS_EDITABLE" = true ]; then
  log_success "deepagents-code${NEW_VERSION:+ ${NEW_VERSION}} reinstalled from PyPI."
elif [ -z "$PRE_VERSION" ]; then
  log_success "deepagents-code${NEW_VERSION:+ ${NEW_VERSION}} installed."
elif [ -n "$NEW_VERSION" ] && [ "$PRE_VERSION" = "$NEW_VERSION" ]; then
  # Same app version, but uv may have refreshed transitive deps (security or
  # compat bumps). The final status line is the user-facing summary, so a flat
  # "already up to date" would contradict the package diff printed just above
  # (and, in non-verbose mode where an addition-only diff is suppressed, hide
  # the dep move entirely). UV_REPORTED_PACKAGE_CHANGES (set far above) is the
  # signal that the reinstall actually moved packages.
  if [ "$UV_REPORTED_PACKAGE_CHANGES" = true ]; then
    # INSTALL_LOG_DISPLAY is empty exactly when no log was written, so the
    # `:+` suffix appends the pointer only when there's a log to point at.
    log_success "deepagents-code ${NEW_VERSION} was already up to date; dependencies were updated.${INSTALL_LOG_DISPLAY:+ Details: ${INSTALL_LOG_DISPLAY}}"
  else
    log_success "deepagents-code ${NEW_VERSION} already up to date."
  fi
elif [ -n "$NEW_VERSION" ]; then
  log_success "deepagents-code updated: ${PRE_VERSION} → ${NEW_VERSION}."
else
  log_success "deepagents-code installed."
fi

if [ "$VERBOSE" = "1" ] && [ -n "$DCODE_BIN_DISPLAY" ]; then
  printf "  Location: %s\n" "$DCODE_BIN_DISPLAY"
fi

if [ "$VERIFY_OK" = true ]; then
  # The prior log_success already named the installed/updated version, so the
  # "Verified" line is redundant for the common case — gate it behind VERBOSE.
  # The empty-output warning stays unconditional: it signals a broken install.
  if [ -z "$NEW_VERSION" ] || [ "$PRE_VERSION" != "$NEW_VERSION" ] || [ "$IS_EDITABLE" = true ]; then
    VERIFY_FIRST=$(printf '%s\n' "$VERIFY_OUTPUT" | head -1)
    if [ -z "$VERIFY_FIRST" ]; then
      log_warn "${DCODE_NAME} -v exited 0 but produced no output; installation may be incomplete."
    elif [ "$VERBOSE" = "1" ]; then
      log_success "Verified: ${DCODE_NAME} ${VERIFY_FIRST}"
    fi
  fi
elif [ -n "$DCODE_BIN" ]; then
  log_warn "${DCODE_NAME} binary found but '${DCODE_NAME} -v' failed:"
  log_warn "  ${VERIFY_OUTPUT}"
  log_warn "The installation may be broken. Try running: ${DCODE_NAME} -v"
else
  log_warn "dcode (or deepagents-code) command not found in PATH. Restart your shell or run:"
  log_warn "  source ~/.zshrc   # (or ~/.bashrc)"
fi

# The binary verified via its absolute path but isn't on the current shell's
# PATH (typical right after a fresh `uv tool install`): typing `dcode` won't
# work until the shell picks up ~/.local/bin. Instead of just telling the user
# to restart their shell, try to fix the PATH now — symlink into an existing
# PATH dir, or add ~/.local/bin to the shell profile — so the binary is
# immediately usable in a new terminal without manual configuration.
if [ "$VERIFY_OK" = true ] && [ "$DCODE_ON_PATH" = false ] && [ -n "$DCODE_BIN" ]; then
  if ! ensure_path_setup "$DCODE_NAME" "$DCODE_BIN"; then
    # ensure_path_setup already printed a specific warning; add the fallback.
    log_warn "  Restart your shell, or run:"
    if [ -f "${HOME}/.local/bin/env" ]; then
      log_warn "  source ~/.local/bin/env"
    else
      log_warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Optional tools — ripgrep
# ---------------------------------------------------------------------------

# Pre-check: verify sudo is usable before running sudo commands.
# Returns 0 if sudo is available (cached or passwordless), 1 otherwise.
check_sudo() {
  if ! command -v sudo >/dev/null 2>&1; then
    return 1
  fi
  # -v -n: validate cached credentials, non-interactive (no password prompt)
  if sudo -v -n 2>/dev/null; then
    return 0
  fi
  # Interactive: warn and let sudo prompt normally
  if [ "$IS_INTERACTIVE" = true ]; then
    log_warn "sudo may prompt for your password."
    return 0
  fi
  return 1
}

install_ripgrep_via_pkg() {
  case "$OS" in
    macos)
      if command -v brew >/dev/null 2>&1; then
        log_info "Installing ripgrep via Homebrew (this may take a moment)..."
        if HOMEBREW_NO_AUTO_UPDATE=1 brew install ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      fi
      if command -v port >/dev/null 2>&1 && check_sudo; then
        log_info "Installing ripgrep via MacPorts..."
        if sudo port install ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      fi
      ;;
    linux)
      if command -v apt-get >/dev/null 2>&1 && check_sudo; then
        log_info "Installing ripgrep via apt-get..."
        if sudo apt-get install -y ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      elif command -v dnf >/dev/null 2>&1 && check_sudo; then
        log_info "Installing ripgrep via dnf..."
        if sudo dnf install -y ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      elif command -v pacman >/dev/null 2>&1 && check_sudo; then
        log_info "Installing ripgrep via pacman..."
        if sudo pacman -S --noconfirm ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      elif command -v zypper >/dev/null 2>&1 && check_sudo; then
        log_info "Installing ripgrep via zypper..."
        if sudo zypper install -y ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      elif command -v apk >/dev/null 2>&1 && check_sudo; then
        log_info "Installing ripgrep via apk..."
        if sudo apk add ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      elif command -v nix-env >/dev/null 2>&1; then
        log_info "Installing ripgrep via nix..."
        if nix-env -iA nixpkgs.ripgrep; then
          command -v rg >/dev/null 2>&1 && return 0
        fi
      fi
      ;;
  esac
  return 1
}

install_ripgrep_via_cargo() {
  if command -v cargo >/dev/null 2>&1; then
    log_info "Installing ripgrep via cargo (no sudo needed)..."
    if cargo install ripgrep; then
      fix_owner "${HOME}/.cargo"
      command -v rg >/dev/null 2>&1 && return 0
      log_warn "cargo install succeeded but rg not found in PATH."
    fi
  fi
  return 1
}

ripgrep_manual_hint() {
  log_warn "ripgrep is not installed; the grep tool will use a slower fallback."
  case "$OS" in
    macos)  log_warn "  Install: brew install ripgrep" ;;
    *)      log_warn "  Install: https://github.com/BurntSushi/ripgrep#installation" ;;
  esac
}

ripgrep_managed_failed() {
  log_warn "Managed ripgrep setup did not complete; the grep tool will use a slower fallback."
  ripgrep_manual_hint
}

if [ "$SKIP_OPTIONAL" != "1" ]; then
  if [ "$RIPGREP_INSTALLER" = "managed" ] && [ "$VERIFY_OK" = true ] && [ -n "$DCODE_BIN" ]; then
    # Eager, non-prompting managed install through the freshly installed binary
    # — the same pinned, SHA-256-verified path dcode uses on first run
    # (downloads into ~/.deepagents/bin, no sudo). Doing it here removes the
    # first-run download latency. The binary reuses a system `rg` already on
    # PATH and honors DEEPAGENTS_CODE_OFFLINE and
    # DEEPAGENTS_CODE_RIPGREP_INSTALLER=system. Routine output stays behind
    # verbose mode because most users do not need ripgrep setup details.
    if [ "$VERBOSE" = "1" ]; then
      echo ""
      log_info "Setting up ripgrep..."
      if "$DCODE_BIN" tools install; then
        fix_owner "${HOME}/.deepagents/bin"
      else
        ripgrep_managed_failed
      fi
    else
      # Quiet path: capture setup output and surface it only on failure, so a
      # broken install stays debuggable without noise in the common case.
      ripgrep_setup_out=$(mktemp 2>/dev/null) || ripgrep_setup_out="/tmp/deepagents-ripgrep-setup.$$.out"
      register_temp "$ripgrep_setup_out"
      if "$DCODE_BIN" tools install >"$ripgrep_setup_out" 2>&1; then
        fix_owner "${HOME}/.deepagents/bin"
      else
        echo ""
        cat "$ripgrep_setup_out" >&2 2>/dev/null || true
        ripgrep_managed_failed
      fi
      rm -f "$ripgrep_setup_out"
    fi
  elif command -v rg >/dev/null 2>&1; then
    if [ "$VERBOSE" = "1" ]; then
      echo ""
      log_info "Checking optional tools..."
      rg_version=$(rg --version 2>/dev/null | head -1 | awk '{print $2}') || rg_version="(version unknown)"
      log_success "ripgrep ${rg_version} found"
    fi
  else
    echo ""
    log_warn "ripgrep not found — recommended for faster file search."

    installed=false
    if prompt_yn "  Install ripgrep?"; then
      if install_ripgrep_via_pkg; then
        installed=true
      elif install_ripgrep_via_cargo; then
        installed=true
      fi

      if [ "$installed" = true ]; then
        log_success "ripgrep installed."
      else
        log_error "Automatic install failed."
        ripgrep_manual_hint
      fi
    else
      ripgrep_manual_hint
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Done — footer wording depends on what changed:
#   - same app version + dependency changes → "Dependencies updated"
#   - already up to date                    → "Already installed"
#   - fresh install / upgrade / editable→PyPI swap → "Setup complete"
# ---------------------------------------------------------------------------
if [ "$IS_EDITABLE" = false ] && [ -n "$PRE_VERSION" ] && [ -n "$NEW_VERSION" ] \
  && [ "$PRE_VERSION" = "$NEW_VERSION" ] && [ "$UV_REPORTED_PACKAGE_CHANGES" = true ]; then
  footer_msg="Dependencies updated."
elif [ "$IS_EDITABLE" = false ] && [ -n "$PRE_VERSION" ] && [ -n "$NEW_VERSION" ] \
  && [ "$PRE_VERSION" = "$NEW_VERSION" ]; then
  footer_msg="Already installed."
else
  footer_msg="Setup complete."
fi
echo ""
# shellcheck disable=SC2059
printf "${GREEN}✔${NC} %s Run: ${BOLD}dcode${NC}\n" "$footer_msg"
echo "  Docs: https://docs.langchain.com/deepagents-code"
