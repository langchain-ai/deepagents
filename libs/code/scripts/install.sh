#!/usr/bin/env bash
# Install deepagents-code.
#
# Usage:
#   curl -LsSf https://langch.in/dcode | bash
#   curl -LsSf https://langch.in/dcode | bash -s -- VERSION
#
# Install an exact pre-release version:
#   curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_VERSION="0.1.0rc1" bash
#   curl -LsSf https://langch.in/dcode | bash -s -- 0.1.0rc1
#
# Override uv's pre-release strategy when resolving the latest version:
#   curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_PRERELEASE="allow" bash
#
# Options:
#   --help, -h     Show this help message and exit
#   --version, -v  Print installer version and exit
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
#       Standalone integrations: media, quickjs
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
#   DEEPAGENTS_CODE_NO_MODIFY_PATH — set to 1 to skip PATH setup entirely
#     (no profile edits, no symlinks). The binary is still installed and
#     verified; add the tool bin dir to PATH yourself, or run it by its
#     absolute path. Intended for version-managed dotfiles and MDM fleets.
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
#   Multi-profile shell coverage, the fish conf.d file, ZDOTDIR resolution,
#   and the no-modify-PATH escape hatch adapted from muse
#   (https://dev.meta.ai/install.sh).

set -euo pipefail

# ---------------------------------------------------------------------------
# CLI flags — --help / --version short-circuit before any install work
# ---------------------------------------------------------------------------
INSTALLER_VERSION="deepagents-code installer 1.0"

print_help() {
  cat <<'HELP'
Install deepagents-code.

Usage:
  curl -LsSf https://langch.in/dcode | bash
  curl -LsSf https://langch.in/dcode | bash -s -- [options]
  curl -LsSf https://langch.in/dcode | bash -s -- VERSION

Options:
  --help, -h        Show this help message and exit
  --version, -v     Print installer version and exit

Target:
  VERSION           Install an exact version, e.g. 0.1.0rc1

Environment variables:
  DEEPAGENTS_CODE_EXTRAS — comma-separated pip extras, e.g. "ollama",
    "ollama,groq", or "daytona". Valid extras (see pyproject.toml for the
    authoritative list):
      Model providers: anthropic, baseten, bedrock, cohere, deepseek,
        fireworks, google-genai, groq, huggingface, ibm, litellm, mistralai,
        nvidia, ollama, openai, openrouter, perplexity, together, vertex, xai,
        all-providers
      Sandbox providers: agentcore, daytona, modal, runloop, vercel,
        all-sandboxes
      Standalone integrations: media, quickjs
  DEEPAGENTS_CODE_VERSION — exact version to install, e.g. "0.1.0rc1"
    (mutually exclusive with DEEPAGENTS_CODE_PRERELEASE)
  DEEPAGENTS_CODE_PRERELEASE — uv pre-release strategy applied when
    resolving the latest version: disallow, allow, if-necessary, explicit,
    or if-necessary-or-explicit (default: allow; explicitly setting it is
    mutually exclusive with DEEPAGENTS_CODE_VERSION)
  DEEPAGENTS_CODE_PYTHON — Python version to use (default: 3.13)
  DEEPAGENTS_CODE_YES — set to 1 to accept an available update without
    prompting (assume "yes")
  DEEPAGENTS_CODE_SKIP_OPTIONAL — set to 1 to skip optional tool checks
  DEEPAGENTS_CODE_RIPGREP_INSTALLER — how to provision ripgrep:
    "managed" (default) eagerly installs the pinned, SHA-256-verified binary
    into ~/.deepagents/bin (no sudo) via `dcode tools install`; "system"
    keeps the interactive package-manager install (brew/apt/cargo/...). Set
    DEEPAGENTS_CODE_OFFLINE=1 to skip the managed download entirely.
  DEEPAGENTS_CODE_SKIP_XCODE_CHECK — set to 1 to bypass the macOS Xcode
    Command Line Tools preflight check
  DEEPAGENTS_CODE_NO_MODIFY_PATH — set to 1 to skip PATH setup entirely
    (no profile edits, no symlinks)
  DEEPAGENTS_CODE_VERBOSE — set to 1 to show uv's raw stderr and additional
    status lines
  UV_BIN — path to uv binary (auto-detected if unset)

For full documentation: https://docs.langchain.com/deepagents-code
HELP
}

POSITIONAL_TARGET=""
for _arg in "$@"; do
  case "$_arg" in
    --help|-h)
      print_help
      exit 0
      ;;
    --version|-v)
      printf '%s\n' "$INSTALLER_VERSION"
      exit 0
      ;;
    -*)
      # Reject unknown flags (single- or double-dash) instead of silently
      # ignoring them, so a typo (e.g. --verison, -V) surfaces as an error
      # rather than a silent full install. Matching every dash-led token here
      # also keeps such typos out of the positional-target arm below, where a
      # leading dash would otherwise be reported as an "invalid version".
      # log_* helpers aren't defined yet at this point, so write plainly.
      printf 'Unrecognized argument: %s\n' "$_arg" >&2
      printf 'Run with --help to see available options.\n' >&2
      exit 2
      ;;
    *)
      if [ -n "$POSITIONAL_TARGET" ]; then
        printf 'Only one target is allowed. Got both %s and %s.\n' "$POSITIONAL_TARGET" "$_arg" >&2
        printf 'Run with --help to see available options.\n' >&2
        exit 2
      fi
      # Same validation (and rationale) as the DEEPAGENTS_CODE_VERSION check
      # further down: require a leading alphanumeric and a class free of shell
      # metacharacters, so the value is a version, not a smuggled option, and is
      # safe to interpolate into the single argv token passed to uv.
      if [[ ! "$_arg" =~ ^[A-Za-z0-9][A-Za-z0-9_.!+-]*$ ]]; then
        printf 'Invalid version target: %s\n' "$_arg" >&2
        printf 'Use an exact version like 0.1.0rc1.\n' >&2
        exit 2
      fi
      POSITIONAL_TARGET="$_arg"
      ;;
  esac
done

# Registry of temp files to clean up on exit or interrupt. Functions that
# create tempfiles append their paths here; cleanup_on_signal removes them all.
TEMP_FILES=()
INSTALL_LOCK_KIND=""
INSTALL_LOCK_DIR=""
INSTALL_LOCK_TOKEN=""
INSTALL_LOCK_STALE_ID=""
INSTALL_LOCK_RECLAIM_DIR=""
INSTALL_LOCK_RECLAIM_TOKEN=""
# How old a mkdir-based lock (dead/unknown holder) must be before it's treated
# as abandoned and reclaimed. 10 min comfortably exceeds a normal install.
INSTALL_LOCK_STALE_AFTER_SECS=600
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

is_linux_os() {
  [ "${OS:-}" = "linux" ] || [ "$(uname -s 2>/dev/null)" = "Linux" ]
}

restore_terminal_after_signal() {
  local exit_code="$1"
  if [ "$exit_code" -ge 128 ] && [ -t 0 ]; then
    stty sane 2>/dev/null || true
  fi
}

log_signal_failure_hint() {
  local exit_code="$1"
  if [ "${SIGNAL_FAILURE_HINT_SHOWN:-false}" = true ]; then
    return 0
  fi
  if [ "$exit_code" -eq 137 ] && is_linux_os; then
    log_error "Installation was killed before it could finish (exit code 137). This usually means the system ran out of memory."
    log_error "Free up memory or use a machine with more memory, then run this installer again."
    SIGNAL_FAILURE_HINT_SHOWN=true
  elif [ "$exit_code" -ge 128 ]; then
    log_error "Installation was killed before it could finish (exit code ${exit_code})."
    SIGNAL_FAILURE_HINT_SHOWN=true
  fi
}

# ---------------------------------------------------------------------------
# Exit / interrupt traps — ensures the user always sees an actionable message
# on failure and temp files are cleaned up on Ctrl-C / SIGTERM.
# ---------------------------------------------------------------------------
cleanup_on_signal() {
  local exit_code=$?
  cleanup_temp_files
  if declare -F release_install_lock >/dev/null 2>&1; then
    release_install_lock
  fi
  if [ $exit_code -ne 0 ]; then
    restore_terminal_after_signal "$exit_code"
    echo "" >&2
    log_signal_failure_hint "$exit_code"
    log_error "Installation failed (exit code ${exit_code}). See errors above."
    log_error "For help, visit: https://docs.langchain.com/deepagents-code"
  fi
}
trap cleanup_on_signal EXIT

cleanup_on_interrupt() {
  # Default to SIGINT. This runs as a signal handler, so an unbound $1 under
  # `set -u` would abort partway through and let the EXIT trap print the
  # contradictory "Installation failed" line that disarming it exists to
  # prevent — the interrupt path must never depend on the caller's argument.
  local sig="${1:-2}"
  local exit_code=$((128 + sig))
  # Disarm the EXIT trap first: exiting from here would otherwise also fire
  # cleanup_on_signal, appending a contradictory "Installation failed" message
  # after the friendly interrupt notice below. Temp files are still cleaned up
  # explicitly here, so nothing leaks despite the disarm.
  trap - EXIT
  if [ -t 0 ]; then
    stty sane 2>/dev/null || true
  fi
  echo "" >&2
  log_warn "Installation interrupted."
  cleanup_temp_files
  if declare -F release_install_lock >/dev/null 2>&1; then
    release_install_lock
  fi
  # Exit with the conventional 128+signal code so callers (CI, wrappers) can
  # distinguish an interrupted install from an ordinary failure.
  exit "$exit_code"
}
trap 'cleanup_on_interrupt 2' INT
trap 'cleanup_on_interrupt 15' TERM
trap 'cleanup_on_interrupt 1' HUP

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
    fix_file_owner() { :; }
  else
    fix_owner() {
      if ! chown -R "$TARGET_USER" "$@" 2>&1; then
        log_warn "Could not fix ownership of $* for user ${TARGET_USER}."
      fi
    }
    fix_file_owner() {
      local path
      for path in "$@"; do
        if { [ -e "$path" ] || [ -L "$path" ]; } && ! chown -h "$TARGET_USER" "$path" 2>&1; then
          log_warn "Could not fix ownership of $path for user ${TARGET_USER}."
        fi
      done
    }
  fi
else
  fix_owner() { :; }
  fix_file_owner() { :; }
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
    # `-m` with `-p` only sets the mode on the deepest dir (SC2174); any parents
    # -p creates keep the umask default. Create, then chmod the target itself so
    # 0700 is reliably applied to cache_root.
    mkdir -p "$cache_root" 2>/dev/null || return 1
    chmod 700 "$cache_root" 2>/dev/null || return 1
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

# Epoch mtime of the lock directory, used as a fallback reference time when the
# started_at metadata is missing. Portable across BSD (macOS) and GNU stat.
lock_dir_mtime() {
  local dir="${1:-$INSTALL_LOCK_DIR}"
  local mtime
  mtime="$(stat -f %m "$dir" 2>/dev/null || true)"
  case "$mtime" in
    ''|*[!0-9]*) ;;
    *) printf '%s' "$mtime"; return 0 ;;
  esac

  mtime="$(stat -c %Y "$dir" 2>/dev/null || true)"
  case "$mtime" in
    ''|*[!0-9]*) printf '0' ;;
    *) printf '%s' "$mtime" ;;
  esac
}

# A fingerprint of the lock instance currently at the canonical path, used to
# detect whether it is still the same lock a prior check inspected (see the
# reclaim path in acquire_install_lock). Prints four newline-separated fields —
# token, pid, started_at, dir mtime — compared only by string equality, so the
# field set and order are load-bearing. Returns 1 (empty output) when the lock
# dir is gone; an unreadable field reads as empty, i.e. is treated as absent.
install_lock_identity() {
  [ -d "$INSTALL_LOCK_DIR" ] || return 1

  local token pid started_at mtime
  token="$(cat "$INSTALL_LOCK_DIR/token" 2>/dev/null || true)"
  pid="$(cat "$INSTALL_LOCK_DIR/pid" 2>/dev/null || true)"
  started_at="$(cat "$INSTALL_LOCK_DIR/started_at" 2>/dev/null || true)"
  mtime="$(lock_dir_mtime)"
  printf '%s\n%s\n%s\n%s' "$token" "$pid" "$started_at" "$mtime"
}

# Decide whether an existing mkdir-based lock may be reclaimed. Only ever called
# in the mkdir fallback path (kernel advisory locks self-release on holder death,
# so they need no staleness heuristic). On a "stale" result, also sets the global
# INSTALL_LOCK_STALE_ID to the lock's identity fingerprint (consumed by the
# reclaim path in acquire_install_lock); clears it otherwise. Must be called in a
# condition context (`if install_lock_is_stale`) so `set -e` is suppressed for
# its body: the bare `[ -n ... ]` test near the end returns non-zero on the
# not-stale branch and would otherwise abort the script.
install_lock_is_stale() {
  INSTALL_LOCK_STALE_ID=""
  [ -d "$INSTALL_LOCK_DIR" ] || return 1

  local pid started_at now
  pid="$(cat "$INSTALL_LOCK_DIR/pid" 2>/dev/null || true)"
  started_at="$(cat "$INSTALL_LOCK_DIR/started_at" 2>/dev/null || true)"
  now="$(date +%s 2>/dev/null || printf '0')"

  # A live owner is authoritative: never reclaim a lock whose PID is running.
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  # When started_at is missing or not yet written, fall back to the lock dir's
  # mtime. This covers the window between `mkdir` winning and the metadata being
  # written by the new owner: treating that window as "stale" would let a racing
  # installer delete a lock that was just acquired, so age it out via mtime
  # instead of reclaiming it on sight.
  case "$started_at" in
    ''|*[!0-9]*) started_at="$(lock_dir_mtime)" ;;
  esac
  case "$started_at" in
    ''|*[!0-9]*) started_at=0 ;;
  esac

  # Without a usable reference or current time we cannot prove the lock is old.
  # Be conservative and keep waiting rather than risk deleting a live lock; a
  # working host always has these, so this only guards pathological environments.
  if [ "$started_at" -eq 0 ] || [ "$now" -eq 0 ]; then
    return 1
  fi

  if [ $((now - started_at)) -ge "$INSTALL_LOCK_STALE_AFTER_SECS" ]; then
    # Capture the identity for the reclaim guard. If the lock dir vanished
    # between the age check and here, the fingerprint is empty; the bare test
    # then returns non-zero, so `return` reports "not stale" (see header note on
    # why this is safe under `set -e`).
    INSTALL_LOCK_STALE_ID="$(install_lock_identity 2>/dev/null || true)"
    [ -n "$INSTALL_LOCK_STALE_ID" ]
    return
  fi
  return 1
}

install_lock_reclaim_guard_is_stale() {
  local started_at
  local now

  started_at="$(cat "$INSTALL_LOCK_RECLAIM_DIR/started_at" 2>/dev/null || true)"
  now="$(date +%s 2>/dev/null || printf '0')"

  case "$started_at" in
    ''|*[!0-9]*) started_at="$(lock_dir_mtime "$INSTALL_LOCK_RECLAIM_DIR")" ;;
  esac
  case "$started_at" in
    ''|*[!0-9]*) started_at=0 ;;
  esac

  if [ "$started_at" -eq 0 ] || [ "$now" -eq 0 ]; then
    return 1
  fi

  [ $((now - started_at)) -ge "$INSTALL_LOCK_STALE_AFTER_SECS" ]
}

wait_for_install_lock_reclaim_guard() {
  if [ ! -d "$INSTALL_LOCK_RECLAIM_DIR" ]; then
    return 0
  fi

  if install_lock_reclaim_guard_is_stale; then
    log_error "Installer lock reclaim is stuck at $INSTALL_LOCK_RECLAIM_DIR."
    log_error "Remove it manually, then retry."
    exit 1
  fi

  sleep 1
  return 1
}

acquire_install_lock_reclaim_guard() {
  local token

  token="$$:$(date +%s 2>/dev/null || printf '0'):${RANDOM:-0}"
  if ! mkdir "$INSTALL_LOCK_RECLAIM_DIR" 2>/dev/null; then
    if [ -d "$INSTALL_LOCK_RECLAIM_DIR" ]; then
      return 1
    fi
    log_error "Cannot reclaim stale installer lock at $INSTALL_LOCK_DIR."
    log_error "Cannot create reclaim guard at $INSTALL_LOCK_RECLAIM_DIR."
    exit 1
  fi

  if ! printf '%s\n' "$token" >"$INSTALL_LOCK_RECLAIM_DIR/token" 2>/dev/null; then
    rm -rf "$INSTALL_LOCK_RECLAIM_DIR" 2>/dev/null || true
    log_error "Cannot reclaim stale installer lock at $INSTALL_LOCK_DIR."
    log_error "Cannot write reclaim guard metadata at $INSTALL_LOCK_RECLAIM_DIR."
    exit 1
  fi

  INSTALL_LOCK_RECLAIM_TOKEN="$token"
  printf '%s\n' "$$" >"$INSTALL_LOCK_RECLAIM_DIR/pid" 2>/dev/null || true
  date +%s >"$INSTALL_LOCK_RECLAIM_DIR/started_at" 2>/dev/null || true
  return 0
}

release_install_lock_reclaim_guard() {
  # Token-guarded like release_install_lock: only drop the guard if it is still
  # ours (see that function for why an unreadable token errs toward keeping it).
  if [ -n "${INSTALL_LOCK_RECLAIM_TOKEN:-}" ] && \
    [ "$(cat "$INSTALL_LOCK_RECLAIM_DIR/token" 2>/dev/null || true)" = "$INSTALL_LOCK_RECLAIM_TOKEN" ]; then
    rm -rf "$INSTALL_LOCK_RECLAIM_DIR" 2>/dev/null || true
  fi
  INSTALL_LOCK_RECLAIM_TOKEN=""
}

# Serialize concurrent installs (racing `curl | bash` runs corrupting a shared
# uv tool dir). Use an atomic mkdir lock dir with a PID + timestamp so a crashed
# holder's lock can be aged out (see install_lock_is_stale). Avoid shell
# redirection to a lock file here: when the installer runs as root and HOME is
# user-writable, opening ~/.deepagents/install.lock would follow a symlink before
# any post-open validation can run.
acquire_install_lock() {
  local lock_root="$HOME/.deepagents"
  mkdir -p "$lock_root"
  fix_owner "$lock_root"

  INSTALL_LOCK_DIR="$lock_root/install.lock.d"
  INSTALL_LOCK_RECLAIM_DIR="$lock_root/install.lock.reclaim.d"

  while true; do
    wait_for_install_lock_reclaim_guard || continue

    if mkdir "$INSTALL_LOCK_DIR" 2>/dev/null; then
      break
    fi

    if install_lock_is_stale; then
      local _stale_id="${INSTALL_LOCK_STALE_ID:-}"
      if [ -z "$_stale_id" ] || ! acquire_install_lock_reclaim_guard; then
        continue
      fi
      if [ "$(install_lock_identity 2>/dev/null || true)" != "$_stale_id" ]; then
        release_install_lock_reclaim_guard
        continue
      fi

      log_warn "Removing stale installer lock at $INSTALL_LOCK_DIR"
      # Only reclaim the exact lock instance that was inspected as stale, so this
      # rename can never move a fresh owner's lock aside. Two mechanisms cover
      # the window: the identity re-check above rejects a lock that was already
      # replaced before we took the reclaim guard, and the guard then blocks
      # peers (via wait_for_install_lock_reclaim_guard) from creating a fresh
      # lock at the canonical path until this rename completes.
      local _stale_reclaim="${INSTALL_LOCK_DIR}.reclaim.$$"
      if mv "$INSTALL_LOCK_DIR" "$_stale_reclaim" 2>/dev/null; then
        rm -rf "$_stale_reclaim" 2>/dev/null || true
        release_install_lock_reclaim_guard
      elif [ "$(install_lock_identity 2>/dev/null || true)" != "$_stale_id" ]; then
        release_install_lock_reclaim_guard
        continue
      else
        release_install_lock_reclaim_guard
        # The stale lock can be neither renamed nor removed (typically it is
        # owned by another user). Fail loudly rather than spin: `continue` skips
        # the `sleep` below, so silently swallowing this error would busy-loop
        # and spam the warning above forever.
        log_error "Cannot reclaim stale installer lock at $INSTALL_LOCK_DIR."
        log_error "Remove it manually or rerun as its owner, then retry."
        exit 1
      fi
      continue
    fi
    sleep 1
  done

  INSTALL_LOCK_TOKEN="$$:$(date +%s 2>/dev/null || printf '0'):${RANDOM:-0}"
  if ! printf '%s\n' "$INSTALL_LOCK_TOKEN" >"$INSTALL_LOCK_DIR/token" 2>/dev/null; then
    rm -rf "$INSTALL_LOCK_DIR" 2>/dev/null || true
    log_error "Cannot write installer lock metadata at $INSTALL_LOCK_DIR."
    exit 1
  fi
  printf '%s\n' "$$" >"$INSTALL_LOCK_DIR/pid"
  date +%s >"$INSTALL_LOCK_DIR/started_at" 2>/dev/null || true
  fix_owner "$INSTALL_LOCK_DIR"
  INSTALL_LOCK_KIND="mkdir"
}

release_install_lock() {
  case "${INSTALL_LOCK_KIND:-}" in
    mkdir)
      # Remove the lock only while the on-disk token is still ours. A clean read
      # that differs means a reclaimer took over the canonical path — never
      # delete their live lock. An unreadable token is treated the same (skip):
      # we cannot prove ownership, and erring toward a leak is safe because a
      # stale lock ages out via install_lock_is_stale, whereas removing on an
      # unverifiable read could delete a reclaimer's lock. Do not turn this into
      # an unconditional `rm -rf`.
      if [ -n "${INSTALL_LOCK_TOKEN:-}" ] && [ "$(cat "$INSTALL_LOCK_DIR/token" 2>/dev/null || true)" = "$INSTALL_LOCK_TOKEN" ]; then
        rm -rf "$INSTALL_LOCK_DIR" 2>/dev/null || true
      fi
      ;;
  esac
  release_install_lock_reclaim_guard
  INSTALL_LOCK_KIND=""
  INSTALL_LOCK_TOKEN=""
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXTRAS="${DEEPAGENTS_CODE_EXTRAS:-}"
VERSION="${DEEPAGENTS_CODE_VERSION:-}"
PRERELEASE_REQUESTED="${DEEPAGENTS_CODE_PRERELEASE:-}"
if [ -n "$POSITIONAL_TARGET" ]; then
  if [ -n "$VERSION" ]; then
    log_error "Do not combine a positional version with DEEPAGENTS_CODE_VERSION."
    log_error "Use either the version argument, or the environment variable — not both."
    exit 1
  fi
  VERSION="$POSITIONAL_TARGET"
fi
PRERELEASE="${PRERELEASE_REQUESTED:-allow}"
PYTHON_REQUESTED=false
if [[ -n "${DEEPAGENTS_CODE_PYTHON:-}" ]]; then
  PYTHON_REQUESTED=true
fi
PYTHON_VERSION="${DEEPAGENTS_CODE_PYTHON:-3.13}"
SKIP_OPTIONAL="${DEEPAGENTS_CODE_SKIP_OPTIONAL:-0}"
VERBOSE="${DEEPAGENTS_CODE_VERBOSE:-0}"
ASSUME_YES="$(printf '%s' "${DEEPAGENTS_CODE_YES:-0}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
case "$ASSUME_YES" in
  1|true|yes) ASSUME_YES="1" ;;
  *)          ASSUME_YES="0" ;;
esac
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

# Base URL for a version-specific GitHub release, surfaced before the update
# prompt so users can review what changed before agreeing to upgrade. Release
# tags are `deepagents-code==X.Y.Z`; the `==` must be percent-encoded (%3D%3D)
# for the tag URL to resolve.
RELEASE_TAG_URL_BASE="https://github.com/langchain-ai/deepagents/releases/tag/deepagents-code%3D%3D"

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

# BusyBox wget intentionally implements only a small subset of GNU wget's
# long options. Keep the stricter redirect controls when this wget supports
# them, while still allowing minimal Linux systems to download from an HTTPS
# URL with normal TLS certificate validation.
#
# The help text is captured rather than piped straight into grep: BusyBox exits
# 1 from `--help`, and under `set -o pipefail` that failure becomes the
# pipeline's status even when grep matched, so every option would report as
# unsupported. That fails *open* — it silently drops `-S` and disables the
# redirect audit below on exactly the minimal systems this branch exists for.
# `|| true` because a nonzero `--help` is a BusyBox quirk, not "unsupported".
#
# The result is cached: this is probed once per hardening flag per attempt, and
# the defaults keep the function self-contained so it can be extracted and
# exercised on its own.
WGET_HELP_TEXT=""
WGET_HELP_CACHED=false
wget_supports_option() {
  if [ "${WGET_HELP_CACHED:-false}" = false ]; then
    WGET_HELP_TEXT="$(wget --help 2>&1 || true)"
    WGET_HELP_CACHED=true
  fi
  printf '%s\n' "${WGET_HELP_TEXT:-}" | grep -Fq -- "$1"
}

# Download with wget, adding hardening options only when this wget implements
# them. `output` may be `-` to write the response to stdout.
#
# GNU wget has no real equivalent of curl's --proto-redir: `--https-only` is
# documented (and implemented) as "when in recursive mode, only HTTPS links are
# followed", so it does NOT stop a 3xx from downgrading a plain one-shot fetch
# to plaintext. We therefore pass it for what it's worth, but enforce the
# scheme ourselves: the request URL must be HTTPS, and the response headers are
# audited for a non-HTTPS Location so a redirect downgrade fails closed.
wget_download() {
  local output="$1" url="$2" ua="${3:-}" quiet="${4:-false}"
  case "$url" in
    https://*) ;;
    *)
      log_warn "Refusing to download over a non-HTTPS URL: ${url}"
      return 1
      ;;
  esac
  local -a args
  if [ "$quiet" = "true" ]; then
    args=(-qO "$output")
  else
    args=(-nv -O "$output")
  fi
  if [ -n "$ua" ] && wget_supports_option "--header"; then
    args+=("--header=User-Agent: ${ua}")
  fi
  if wget_supports_option "--max-redirect"; then
    args+=(--max-redirect=3)
  fi
  if wget_supports_option "--https-only"; then
    args+=(--https-only)
  fi
  # -S echoes the response headers (including each redirect's Location) to
  # stderr, which is what makes the downgrade audit below possible. BusyBox
  # wget lacks it; there the audit degrades to a no-op, so BusyBox hosts keep
  # working but rely on wget's own behavior for redirect handling.
  local can_audit=false
  if wget_supports_option "-S"; then
    args+=(-S)
    can_audit=true
  else
    # Say so rather than degrading in silence: without -S there is no
    # downgrade check at all on the wget path (--https-only does not stop a
    # one-shot redirect), and a control that is believed active but inert is
    # worse than one the user knows is missing.
    log_warn "This wget cannot report response headers (-S); cannot verify redirects stay on HTTPS."
  fi

  local wget_err rc=0
  wget_err=$(mktemp 2>/dev/null) || return 1
  register_temp "$wget_err"
  wget "${args[@]}" "$url" 2>"$wget_err" || rc=$?
  if [ "$can_audit" = true ] && \
    grep -qiE '^[[:space:]]*Location:[[:space:]]*http:' "$wget_err"; then
    log_warn "Refusing a plaintext HTTP redirect while downloading ${url}."
    rm -f "$wget_err"
    return 1
  fi
  # Forward wget's own diagnostics so the caller's stderr capture still sees
  # the real failure reason; -qO callers asked for silence.
  if [ "$quiet" != "true" ]; then
    cat "$wget_err" >&2
  fi
  rm -f "$wget_err"
  return "$rc"
}

# Download a URL to stdout using the first available working downloader.
# Prefers curl (unless it's a snap install, which has sandbox permission
# issues), then falls back to wget. Prints nothing and returns non-zero if no
# working downloader is available.
# The curl call pins HTTPS for the request and any redirects (--proto /
# --proto-redir) so a hostile proxy or DNS answer can't silently downgrade the
# download to plaintext, and caps redirect chains at 3 so a broken or
# malicious redirect loop fails instead of spinning. wget has no --proto-redir
# equivalent, so wget_download enforces the same property by rejecting
# non-HTTPS URLs and auditing the response headers for a downgrade redirect.
download_to_stdout() {
  local url="$1" ua="${2:-deepagents-code-install}"
  local attempt=1 body="" download_rc=1
  while [ "$attempt" -le 3 ]; do
    if command -v curl >/dev/null 2>&1 && ! is_snap_curl; then
      if body=$(curl -fsSL -H "User-Agent: ${ua}" \
        --proto '=https' --proto-redir '=https' --max-redirs 3 \
        "$url" 2>/dev/null); then
        printf '%s' "$body"
        return 0
      else
        download_rc=$?
      fi
    elif command -v wget >/dev/null 2>&1; then
      if body=$(wget_download - "$url" "$ua" true 2>/dev/null); then
        printf '%s' "$body"
        return 0
      else
        download_rc=$?
      fi
    else
      return 1
    fi
    # 2s then 4s. A 1s/2s backoff is too short to outlast the transient it is
    # meant to absorb (a captive-portal redirect settling, a CDN edge failing
    # over); 6s of total waiting is still well inside a user's patience.
    if [ "$attempt" -lt 3 ]; then
      sleep $((attempt * 2))
    fi
    attempt=$((attempt + 1))
  done
  return "$download_rc"
}

install_uv() {
  # The upstream uv installer is chatty (download progress, install paths,
  # PATH-setup hints). Capture it and surface the output only when debugging
  # or when the install fails — by default it's noise the user doesn't need.
  # This same tempfile also captures the downloader's stderr, so a failed
  # download surfaces curl/wget's own error (DNS, TLS, HTTP status) instead of
  # a generic message; the installer's stdout/stderr overwrites it afterward.
  local uv_install_out uv_install_rc=0
  uv_install_out=$(mktemp 2>/dev/null) || {
    log_error "mktemp is required to create a secure temp file."
    exit 1
  }
  register_temp "$uv_install_out"

  # Download the installer to a tempfile first instead of piping curl straight
  # to sh, so we can verify the first line is a shell shebang before executing.
  # A transparent proxy or captive portal returning 200 with HTML would
  # otherwise pipe straight into sh with unpredictable results. curl's `-f`
  # catches HTTP errors, but a 200-with-HTML response passes that check.
  local uv_script
  uv_script=$(mktemp 2>/dev/null) || {
    log_error "mktemp is required to create a secure temp file."
    exit 1
  }
  register_temp "$uv_script"

  # Capture the downloader's stderr (2>"$uv_install_out") rather than discarding
  # it: on failure it holds the actionable cause (curl: (6) Could not resolve
  # host, SSL errors, HTTP status), which the failure branch below surfaces.
  # curl -sS and wget -nv stay quiet on success, so this adds no noise then.
  local attempt
  if command -v curl >/dev/null 2>&1 && ! is_snap_curl; then
    uv_install_rc=1
    for attempt in 1 2 3; do
      : >"$uv_install_out"
      if curl -fsSL https://astral.sh/uv/install.sh -o "$uv_script" \
        --proto '=https' --proto-redir '=https' --max-redirs 3 \
        2>"$uv_install_out"; then
        uv_install_rc=0
        break
      else
        uv_install_rc=$?
      fi
      if [ "$attempt" -lt 3 ]; then
        sleep $((attempt * 2))
      fi
    done
  elif command -v wget >/dev/null 2>&1; then
    uv_install_rc=1
    for attempt in 1 2 3; do
      : >"$uv_install_out"
      if wget_download "$uv_script" https://astral.sh/uv/install.sh "" false \
        2>"$uv_install_out"; then
        uv_install_rc=0
        break
      else
        uv_install_rc=$?
      fi
      if [ "$attempt" -lt 3 ]; then
        sleep $((attempt * 2))
      fi
    done
  elif is_snap_curl; then
    rm -f "$uv_install_out" "$uv_script"
    log_error "curl is installed as a snap and cannot download files due to sandbox permissions."
    log_error "Please install wget, or reinstall curl with a different package manager (e.g. apt)."
    exit 1
  else
    rm -f "$uv_install_out" "$uv_script"
    log_error "curl or wget is required to install uv."
    exit 1
  fi

  if [ "$uv_install_rc" -ne 0 ]; then
    # Surface the downloader's own error (captured above) before the generic
    # line, so the user sees the real cause and the downloader's exit code.
    if declare -F restore_terminal_after_signal >/dev/null 2>&1; then
      restore_terminal_after_signal "$uv_install_rc"
    fi
    cat "$uv_install_out" >&2
    rm -f "$uv_install_out" "$uv_script"
    if declare -F log_signal_failure_hint >/dev/null 2>&1; then
      log_signal_failure_hint "$uv_install_rc"
    fi
    log_error "Failed to download uv installer (exit ${uv_install_rc}) from https://astral.sh/uv/install.sh"
    log_error "  Try again, or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit "$uv_install_rc"
  fi

  # Verify the downloaded script starts with a shell shebang before executing
  # it. This catches a non-shell response — an HTML error page or JSON from a
  # proxy or captive portal that returned 200 — that would otherwise fail
  # unpredictably when run by sh. It only inspects the first line, so it is a
  # sanity check on the response type, not an integrity guarantee: it won't
  # detect a truncated or tampered body.
  local uv_shebang
  uv_shebang=$(head -1 "$uv_script")
  if ! printf '%s' "$uv_shebang" | grep -qE '^#!.*(sh|bash)'; then
    rm -f "$uv_install_out" "$uv_script"
    log_error "uv installer download does not start with a shell shebang."
    log_error "  The URL may have returned an error page (proxy, captive portal, or outage)."
    log_error "  Try again, or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  # Parse-check the script before executing it. The shebang check above only
  # proves the response is *meant* to be shell; a truncated download (dropped
  # connection mid-body) would still pass that and then fail unpredictably
  # partway through execution. A parse check reads the whole file without
  # running it, so it catches any truncation that leaves unbalanced syntax (an
  # open quote, `if` without `fi`, an unterminated function body). It is not a
  # completeness guarantee: a body cut at a statement boundary still parses.
  #
  # Use the interpreter named in the script's own shebang rather than always
  # `sh`. Upstream currently ships POSIX sh, but if it ever ships bash-only
  # syntax, checking with dash would report a syntax error and send the user
  # chasing a nonexistent truncated download. The same interpreter runs the
  # script below — checking with bash and then executing with sh would let
  # bash-only syntax pass the check and fail at execution instead, which is
  # the confusing outcome this branch exists to avoid.
  local uv_checker="sh"
  case "$uv_shebang" in
    *bash*) command -v bash >/dev/null 2>&1 && uv_checker="bash" ;;
  esac
  if ! "$uv_checker" -n "$uv_script" 2>"$uv_install_out"; then
    cat "$uv_install_out" >&2
    rm -f "$uv_install_out" "$uv_script"
    log_error "uv installer download failed a shell syntax check — it may be truncated."
    log_error "  Try again, or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  "$uv_checker" "$uv_script" >"$uv_install_out" 2>&1 || uv_install_rc=$?
  if [ "$VERBOSE" = "1" ] || [ "$uv_install_rc" -ne 0 ]; then
    cat "$uv_install_out" >&2
  fi
  rm -f "$uv_install_out" "$uv_script"
  if [ "$uv_install_rc" -ne 0 ]; then
    if declare -F restore_terminal_after_signal >/dev/null 2>&1; then
      restore_terminal_after_signal "$uv_install_rc"
    fi
    if declare -F log_signal_failure_hint >/dev/null 2>&1; then
      log_signal_failure_hint "$uv_install_rc"
    fi
    log_error "uv installation failed. See errors above."
    exit "$uv_install_rc"
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
  acquire_install_lock
  UV_BIN_DIR_PREEXISTED=false
  if [ -d "${HOME}/.local/bin" ]; then
    UV_BIN_DIR_PREEXISTED=true
  fi
  log_info "uv not found — installing..."
  install_uv
  if [ "$UV_BIN_DIR_PREEXISTED" = false ]; then
    fix_file_owner "${HOME}/.local/bin"
  fi
  fix_file_owner "${HOME}/.local/bin/uv" "${HOME}/.local/bin/uvx" "${HOME}/.local/bin/env"
  if ! resolve_uv_bin; then
    log_error "uv not found after installation. Restart your shell or add ~/.local/bin to PATH."
    exit 1
  fi
fi

resolve_tool_bin_dir() {
  local dir=""
  if dir=$("$UV_BIN" tool dir --bin 2>/dev/null) && [ -n "$dir" ]; then
    :
  elif [ -n "${XDG_BIN_HOME:-}" ]; then
    dir="$XDG_BIN_HOME"
  elif [ -n "${XDG_DATA_HOME:-}" ]; then
    dir="${XDG_DATA_HOME}/../bin"
  else
    dir="${HOME}/.local/bin"
  fi
  case "$dir" in
    /*) ;;
    *) dir="$(pwd -P)/${dir}" ;;
  esac
  printf '%s\n' "$dir"
}

TOOL_BIN_DIR="$(resolve_tool_bin_dir)"
TOOL_BIN_DIR_DISPLAY="$TOOL_BIN_DIR"
case "$TOOL_BIN_DIR" in
  "$HOME"/*) TOOL_BIN_DIR_DISPLAY="~${TOOL_BIN_DIR#"$HOME"}" ;;
esac
TOOL_BIN_DIR_PREEXISTED=false
if [ -d "$TOOL_BIN_DIR" ]; then
  TOOL_BIN_DIR_PREEXISTED=true
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
PRE_INSTALL_IS_TOOL=false
PRE_INSTALL_ON_PATH=false
if [ "$(id -u)" -ne 0 ]; then
  for candidate in dcode deepagents-code; do
    if [ -x "${TOOL_BIN_DIR}/${candidate}" ]; then
      PRE_VERSION=$("${TOOL_BIN_DIR}/${candidate}" -v 2>/dev/null | head -1 | awk '{print $NF}') || PRE_VERSION=""
      PRE_INSTALL_IS_TOOL=true
      original=$(PATH="$ORIGINAL_PATH" command -v "$candidate" 2>/dev/null || true)
      if [ -n "$original" ] && \
        { [ "$original" = "${TOOL_BIN_DIR}/${candidate}" ] || [ "$original" -ef "${TOOL_BIN_DIR}/${candidate}" ]; }; then
        PRE_INSTALL_ON_PATH=true
      fi
      break
    fi
  done
  if [ "$PRE_INSTALL_IS_TOOL" = false ]; then
    for candidate in dcode deepagents-code; do
      if command -v "$candidate" >/dev/null 2>&1; then
        PRE_VERSION=$("$candidate" -v 2>/dev/null | head -1 | awk '{print $NF}') || PRE_VERSION=""
        break
      fi
    done
  fi
fi

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
  log_info "dcode ${PRE_VERSION} found — checking for updates..."
  LATEST_VERSION=$(fetch_latest_version)
  if [ -z "$LATEST_VERSION" ]; then
    log_warn "Could not determine the latest version from PyPI — continuing with an upgrade attempt."
  elif [ -n "$EXTRAS" ] || [ "$PYTHON_REQUESTED" = true ]; then
    if [ "$LATEST_VERSION" = "$PRE_VERSION" ]; then
      log_info "deepagents-code is already up to date — rebuilding with requested options."
    else
      log_info "Updating deepagents-code ${PRE_VERSION} → ${LATEST_VERSION} with requested options..."
    fi
  elif [ "$LATEST_VERSION" = "$PRE_VERSION" ] && [ "$PRE_INSTALL_ON_PATH" = true ]; then
    log_success "Already up to date!"
    exit 0
  elif [ "$LATEST_VERSION" = "$PRE_VERSION" ] && [ "$PRE_INSTALL_IS_TOOL" = true ]; then
    log_info "deepagents-code ${PRE_VERSION} is current but is not selected on PATH — repairing its install."
  elif [ "$LATEST_VERSION" = "$PRE_VERSION" ]; then
    log_info "deepagents-code ${PRE_VERSION} is current but is outside uv's configured tool bin — installing it there."
  elif [ "$ASSUME_YES" = "1" ]; then
    log_info "Updating deepagents-code ${PRE_VERSION} → ${LATEST_VERSION}..."
  elif can_prompt; then
    log_info "What's new: ${RELEASE_TAG_URL_BASE}${LATEST_VERSION}"
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
  log_info "dcode ${PRE_VERSION} found — checking for updates..."
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
uv_stderr=$(mktemp 2>/dev/null) || {
  log_error "mktemp is required to create a secure temp file."
  exit 1
}
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
if [ -z "$INSTALL_LOCK_KIND" ]; then
  acquire_install_lock
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
  restore_terminal_after_signal "$uv_rc"
  log_signal_failure_hint "$uv_rc"
  log_error "Failed to install ${PACKAGE}. See errors above."
  # The log captured uv's full stderr (copied just above, before this exit), so
  # point the user at it — non-verbose mode trims uv's lines from the terminal
  # and piped `curl | bash` runs lose scrollback.
  if [ -n "$INSTALL_LOG" ]; then
    log_error "Full install log: ${INSTALL_LOG_DISPLAY}"
  fi
  log_error "Common fixes: check your network, try a different Python version (DEEPAGENTS_CODE_PYTHON=3.12), or install manually."
  exit "$uv_rc"
fi
if path_is_under_home "$TOOL_BIN_DIR"; then
  if [ "$TOOL_BIN_DIR_PREEXISTED" = false ]; then
    fix_file_owner "$TOOL_BIN_DIR"
  fi
  fix_file_owner "${TOOL_BIN_DIR}/dcode" "${TOOL_BIN_DIR}/deepagents-code"
fi
if [ -n "$UV_TOOL_DIR" ] && path_is_under_home "${UV_TOOL_DIR}/deepagents-code"; then
  fix_owner "${UV_TOOL_DIR}/deepagents-code"
elif [ -d "${HOME}/.local/share/uv" ]; then
  fix_owner "${HOME}/.local/share/uv"
fi
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
# After `uv tool install`, dcode lands in uv's configured tool bin directory.
# If that directory is not already in PATH, expose the installed binary through
# one of the user's conventional bin directories.
#
# Strategy (symlink-first adapted from Amp's installer,
# https://ampcode.com/install.sh; multi-profile coverage and the fish conf.d
# file adapted from muse's installer, https://dev.meta.ai/install.sh):
#   1. If a common bin dir (~/.local/bin, ~/bin, ~/.bin) is already in PATH,
#      create a symlink there — no profile modification needed.
#   2. Otherwise, create ~/.local/bin, symlink dcode there, then add
#      ~/.local/bin to the user's shell startup files. Every zsh/bash profile
#      that already exists gets the entry (plus the primary one for the
#      current shell), so dcode resolves even if the user later switches
#      shells; fish gets a standalone conf.d/deepagents-code.fish file, which
#      fish auto-sources, instead of a config.fish edit. Prompt interactively
#      before writing; auto-add in non-interactive mode (CI, cron, piped
#      install).
#   3. Skip the whole thing if the binary was already on PATH or uv's env file
#      exposes a binary installed under ~/.local/bin.
#   DEEPAGENTS_CODE_NO_MODIFY_PATH=1 skips steps 1–2 entirely, for users with
#   version-managed dotfiles or MDM-managed PATH.

# Check if a directory was in PATH before the installer sourced any env files.
dir_in_original_path() {
  local check_dir="$1"
  [ -d "$check_dir" ] || return 1
  check_dir=$(cd "$check_dir" 2>/dev/null && pwd) || return 1
  case ":${ORIGINAL_PATH:-}:" in
    *":$check_dir:"*) return 0 ;;
    *) return 1 ;;
  esac
}

paths_are_same_file() {
  [ "$1" = "$2" ] || [ "$1" -ef "$2" ]
}

# Try to symlink the dcode binary into a directory already in PATH. Tries
# ~/.local/bin, ~/bin, and ~/.bin in order. Returns 0 on success.
try_symlink_in_path() {
  local binary_name="$1"
  local binary_path="$2"
  local preferred_dirs=("$HOME/.local/bin" "$HOME/bin" "$HOME/.bin")
  local dir symlink_path
  for dir in "${preferred_dirs[@]}"; do
    if dir_in_original_path "$dir"; then
      mkdir -p "$dir" 2>/dev/null || continue
      symlink_path="$dir/$binary_name"
      if paths_are_same_file "$binary_path" "$symlink_path"; then
        return 0
      fi
      if [ -e "$symlink_path" ] && [ ! -L "$symlink_path" ]; then
        continue
      fi
      # Remove existing symlink if it points elsewhere or is stale
      if [ -L "$symlink_path" ]; then
        rm -f "$symlink_path"
      fi
      if ln -s "$binary_path" "$symlink_path" 2>/dev/null; then
        fix_file_owner "$symlink_path" 2>/dev/null || true
        return 0
      fi
    fi
  done
  return 1
}

# Detect the user's shell and return the profile file + PATH export statement.
# Sets SHELL_PROFILE, PATH_EXPORT, and DETECTED_SHELL as globals.
#
# For fish, SHELL_PROFILE names config.fish only so the reload hint and the
# "add this yourself" message have something to point at. The PATH entry itself
# is never written there: ensure_path_setup writes a standalone
# conf.d/deepagents-code.fish (see fish_conf_file) and never adds config.fish
# as a write candidate.
detect_shell_profile() {
  local default_shell="bash"
  if [ "$OS" = "macos" ]; then
    default_shell="zsh"
  fi
  local shell_name
  shell_name=$(basename "${SHELL:-$default_shell}" 2>/dev/null) || shell_name="$default_shell"
  # Callers must reuse this rather than re-deriving from $SHELL: an empty
  # SHELL would otherwise resolve differently here and there.
  DETECTED_SHELL="${shell_name:-$default_shell}"
  shell_name="$DETECTED_SHELL"
  SHELL_PROFILE=""
  PATH_EXPORT=""
  case "$shell_name" in
    zsh)
      local zdotdir
      zdotdir="$(resolve_zdotdir)"
      SHELL_PROFILE="${zdotdir:-$HOME}/.zshrc"
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
      # Reload hint sources config.fish; the PATH entry itself is written to
      # conf.d/deepagents-code.fish by ensure_path_setup. Honor XDG_CONFIG_HOME
      # here too, or the hint names a file the user doesn't have.
      SHELL_PROFILE="${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"
      # shellcheck disable=SC2016  # single-quoted so $HOME expands at profile source time, not here
      PATH_EXPORT='fish_add_path "$HOME/.local/bin"'
      ;;
    *)
      # Unknown shell — don't modify any profile.
      ;;
  esac
}

# The fish PATH entry lives in its own conf.d file, which fish auto-sources at
# startup. A standalone file keeps the installer from mutating the user's main
# config.fish and makes the entry trivially removable.
fish_conf_file() {
  printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/deepagents-code.fish"
}

# shellcheck disable=SC2016  # single-quoted so $PATH stays literal for the fish file
FISH_PATH_EXPORT='contains -- "$HOME/.local/bin" $PATH; or set -gx PATH "$HOME/.local/bin" $PATH'

# The POSIX export line, used for every startup file that isn't fish's conf.d
# file. Kept separate from PATH_EXPORT because PATH_EXPORT tracks the *current
# shell* (and is empty for unknown shells), while the line written to a given
# file has to match *that file's* syntax — a fish user can still have a
# ~/.zshrc candidate, which must not receive fish syntax.
# shellcheck disable=SC2016  # single-quoted so $HOME/$PATH expand at profile source time
POSIX_PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'

# Check if ~/.local/bin is already referenced in the shell profile's PATH
# config. Matches non-commented lines containing .local/bin in a PATH
# assignment or fish_add_path. Returns 0 if already present.
#
# The alternation also recognizes the un-normalized ~/.local/share/../bin
# spelling (share/.. collapses to .local, so it resolves to the same directory
# as ~/.local/bin). Some tools write that spelling into a profile from
# $XDG_DATA_HOME/../bin; without this we'd fail to see ~/.local/bin as already
# on PATH and append a duplicate entry. This covers the common alias only — it
# is not a full path normalizer, so other exotic spellings still slip through.
local_bin_in_profile() {
  local profile="$1"
  [ -f "$profile" ] || return 1
  grep -v '^[[:space:]]*#' "$profile" 2>/dev/null | grep -qE 'PATH=.*(\.local/bin|\.local/share/\.\./bin)' \
    || grep -v '^[[:space:]]*#' "$profile" 2>/dev/null | grep -qE '(fish_add_path|set[[:space:]].*PATH).*(\.local/bin|\.local/share/\.\./bin)'
}

# Net block-nesting change contributed by one shell line, into
# SHELL_BLOCK_DELTA (a global: returning it via $(...) would fork per line).
# Separators that can abut a keyword are flattened to spaces first, so a
# self-contained `if [ x ]; then y; fi` nets zero rather than opening a block
# that never closes. This is a line-oriented approximation, not a shell parser:
# it does not know about quoting or heredocs. It only has to be right about the
# ordinary conditional wrapper in a hand-written ~/.zshenv.
SHELL_BLOCK_DELTA=0
shell_block_delta() {
  local haystack="$1" word
  SHELL_BLOCK_DELTA=0
  haystack="${haystack//;/ }"
  haystack="${haystack//&/ }"
  haystack="${haystack//|/ }"
  haystack="${haystack//(/ }"
  haystack="${haystack//)/ }"
  # Unquoted expansion below is deliberate (word splitting); disable globbing so
  # a `*` in the line can't expand against the cwd.
  set -f
  for word in $haystack; do
    case "$word" in
      if|case|while|for|until) SHELL_BLOCK_DELTA=$((SHELL_BLOCK_DELTA + 1)) ;;
      fi|esac|done)            SHELL_BLOCK_DELTA=$((SHELL_BLOCK_DELTA - 1)) ;;
    esac
  done
  set +f
}

# Resolve the directory holding zsh's .zshrc. ZDOTDIR, when set, relocates all
# of zsh's dotfiles; users who keep ~/.zshrc elsewhere (e.g.
# ~/.config/zsh/.zshrc) would otherwise get a stray new ~/.zshrc that zsh never
# reads, plus no PATH entry in the file zsh does read.
# Checks $ZDOTDIR itself first, then parses a ZDOTDIR assignment out of
# ~/.zshenv (the one file zsh always reads from $HOME, and the canonical place
# to set ZDOTDIR). Prints the resolved directory, or nothing when unresolved.
resolve_zdotdir() {
  local line value="" depth=0 scan
  if [ -n "${ZDOTDIR:-}" ]; then
    printf '%s\n' "$ZDOTDIR"
    return 0
  fi
  [ -n "${HOME:-}" ] || return 0
  # `-r`, not just `-f`: a root-owned or 0600 .zshenv belonging to another user
  # passes `-f` but the redirect below then fails, leaving `value` empty and
  # silently sending the zshrc to ~/.zshrc — the exact misplacement this
  # function exists to prevent. Say so instead of guessing.
  if [ -f "$HOME/.zshenv" ] && [ ! -r "$HOME/.zshenv" ]; then
    log_warn "Cannot read ~/.zshenv to check for a ZDOTDIR setting; assuming zsh reads ~/.zshrc."
    return 0
  fi
  [ -f "$HOME/.zshenv" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    scan="${line%%#*}"
    # Only a top-level assignment is authoritative. The common portable-dotfiles
    # idiom guards the relocation:
    #
    #     if [ -d "$HOME/.config/zsh" ]; then export ZDOTDIR="$HOME/.config/zsh"; fi
    #
    # While that directory is absent the guard is false and zsh reads ~/.zshrc.
    # Honoring the assignment anyway would make the caller create the directory,
    # which flips the guard true and strands every later shell in a config dir
    # holding nothing but our PATH block. Assignments nested in any block are
    # therefore skipped; the one-line `[ -d x ] && export ZDOTDIR=y` form is
    # already skipped because such a line doesn't begin with the assignment.
    if [ "$depth" -eq 0 ]; then
      # Strip leading whitespace, then an optional leading `export`.
      while [[ "$line" == [[:space:]]* ]]; do line="${line#?}"; done
      if [[ "$line" == export[[:space:]]* ]]; then
        line="${line#export}"
        while [[ "$line" == [[:space:]]* ]]; do line="${line#?}"; done
      fi
      if [[ "$line" == ZDOTDIR=* ]]; then
        value="${line#ZDOTDIR=}"
      fi
    fi
    shell_block_delta "$scan"
    depth=$((depth + SHELL_BLOCK_DELTA))
    # A stray closer (or a construct this line-oriented scan misreads) must not
    # drive the depth negative and re-arm the top-level branch inside a block.
    [ "$depth" -lt 0 ] && depth=0
  done <"$HOME/.zshenv"
  # Strip surrounding quotes, or a trailing `;`, `# comment`, or whitespace for
  # the unquoted form.
  case "$value" in
    \"*)
      value="${value#\"}"
      value="${value%%\"*}"
      ;;
    \'*)
      value="${value#\'}"
      value="${value%%\'*}"
      ;;
    *)
      value="${value%%;*}"
      value="${value%%#*}"
      value="${value%%[[:space:]]*}"
      ;;
  esac
  # Expand the ~ and $HOME spellings people actually write in .zshenv.
  case "$value" in
    \~)            value="$HOME" ;;
    \~/*)          value="$HOME/${value#\~/}" ;;
    "\$HOME")      value="$HOME" ;;
    "\${HOME}")    value="$HOME" ;;
    "\$HOME"/*)    value="$HOME/${value#\$HOME/}" ;;
    "\${HOME}"/*)  value="$HOME/${value#\$\{HOME\}/}" ;;
  esac
  # Any absolute path is accepted, whether or not it exists yet. A ZDOTDIR=
  # line in ~/.zshenv tells zsh "look here" — and zsh looks only there — so a
  # valid absolute ZDOTDIR is authoritative for where the zshrc must go even
  # when the directory hasn't been created. An existence check would have the
  # failure mode backwards: on a fresh setup with `ZDOTDIR=$HOME/.config/zsh`,
  # rejecting the value writes ~/.zshrc, a file zsh never reads once ZDOTDIR
  # is set, so `dcode` stays off PATH. The caller creates a missing candidate
  # directory before writing, which is safe only because guarded assignments
  # were skipped above — creating a directory an `if [ -d ... ]` is waiting on
  # would silently activate a ZDOTDIR the user has not opted into yet.
  # Remaining limit: the loop takes the last top-level ZDOTDIR= assignment, so
  # a value set from a sourced file or a shell function is still missed. That
  # errs toward ~/.zshrc, which is where zsh looks when the assignment never
  # runs, and ~/.zshrc is added alongside a ZDOTDIR zshrc whenever it exists.
  case "$value" in
    /*) printf '%s\n' "$value" ;;
  esac
  return 0
}

managed_path_block_present() {
  local profile="$1"
  [ -f "$profile" ] || return 1
  grep -F '# >>> deepagents-code installer >>>' "$profile" >/dev/null 2>&1
}

managed_path_block_has_line() {
  local profile="$1" path_export="$2"
  [ -f "$profile" ] || return 1
  grep -F "$path_export" "$profile" >/dev/null 2>&1
}

append_managed_path_block() {
  local profile="$1" path_export="$2"
  {
    echo ""
    echo "# >>> deepagents-code installer >>>"
    echo "$path_export"
    echo "# <<< deepagents-code installer <<<"
  } >>"$profile"
}

# Resolve a symlink to the final non-symlink path in the chain. The result is
# not canonicalized (a relative hop leaves `..` segments in place) and is not
# guaranteed to exist or to be a regular file — nothing here checks that. It is
# absolute only if the input was.
# Read-only; used to pick the directory an atomic temp-file rewrite must live
# in. Chains (a -> b -> c) must be followed to the end: resolving one hop and
# `mv`-ing over it would replace an intermediate link with a regular file, so
# the real dotfile-manager source stays stale and the next restow recreates the
# link, reverting the PATH entry. The hop count is capped to bound symlink
# loops (ELOOP), which we can't detect portably; 40 matches Linux's
# MAXSYMLINKS, and macOS stops at 32, so any chain a kernel would follow fits.
resolve_link_target() {
  local link="$1" target hops=0
  while [ -L "$link" ]; do
    hops=$((hops + 1))
    if [ "$hops" -gt 40 ]; then
      return 1
    fi
    target=$(readlink "$link") || return 1
    case "$target" in
      /*) link="$target" ;;
      *)  link="$(dirname "$link")/$target" ;;
    esac
  done
  printf '%s\n' "$link"
}

rewrite_managed_path_block() {
  local profile="$1" path_export="$2" tmp_profile mode
  if [ -L "$profile" ]; then
    # Dotfile managers (chezmoi, stow, nix home-manager) symlink startup files
    # into a repo. Replacing the link with a regular file means the next
    # `apply`/`restow` silently reverts the PATH entry — after we printed a
    # success message. Write the chain's final target instead, through the same
    # atomic temp-file + mv used below. Writing the target in place would
    # truncate it first, so an interrupted install (we trap INT/TERM/HUP) could
    # leave the dotfile-manager source empty or partial. Resolving the target
    # first puts the temp file in the target's directory, keeping the mv atomic.
    local link_target
    if link_target=$(resolve_link_target "$profile"); then
      log_warn "$profile is a symlink — editing its target instead of replacing the link."
      profile="$link_target"
    else
      log_warn "Could not resolve the symlink target of $profile; skipping."
      return 1
    fi
  fi
  tmp_profile=$(mktemp "$(dirname "$profile")/.deepagents-code-profile.XXXXXX") || return 1
  register_temp "$tmp_profile"

  awk -v begin="# >>> deepagents-code installer >>>" \
    -v end="# <<< deepagents-code installer <<<" \
    -v line="$path_export" '
    BEGIN { in_block = 0; replaced = 0 }
    $0 == begin {
      # Emit the replacement only for the first marker: any accidental
      # duplicate managed blocks are collapsed into this single one.
      if (!replaced) {
        print begin
        print line
        print end
        replaced = 1
      }
      in_block = 1
      next
    }
    in_block {
      if ($0 == end) {
        in_block = 0
      }
      next
    }
    { print }
    END {
      # A begin marker with no matching end marker means the block is malformed.
      # Fail so the caller keeps the original profile (the mv below is skipped)
      # rather than writing back a truncated file.
      if (in_block != 0) {
        exit 1
      }
    }
  ' "$profile" >"$tmp_profile" || return 1
  # mktemp created tmp_profile as 0600; carry over the profile's real perms so
  # an in-place rewrite doesn't silently tighten (e.g.) a 0644 ~/.zshrc.
  mode=$(stat -f '%OLp' "$profile" 2>/dev/null || stat -c '%a' "$profile" 2>/dev/null || true)
  if [ -n "$mode" ]; then
    # Not fatal — the rewrite is still correct — but the user's file quietly
    # becoming 0600 is the kind of change they should hear about.
    chmod "$mode" "$tmp_profile" 2>/dev/null \
      || log_warn "Could not restore mode ${mode} on $(tilde_display "$profile"); it may now be more restrictive."
  fi
  mv "$tmp_profile" "$profile" || return 1
  # `profile` is the final target when the caller supplied a symlink. The
  # caller still fixes ownership on the original symlink, so also restore the
  # target's ownership after this root-created atomic replacement.
  fix_file_owner "$profile"
  # Explicit: without this the function's status is fix_file_owner's, so the
  # day an ownership helper starts propagating chown failures, a rewrite that
  # fully succeeded would report failure.
  return 0
}

# Ensure dcode is on PATH for new shell sessions. Creates symlinks and/or
# modifies shell startup files as needed. Only acts when the binary verified
# but isn't already on the user's original PATH.
# Returns: 0 = nothing more to say — PATH is fixed for the current shell (a
#              symlink in an on-PATH dir), or setup was deliberately skipped
#              and this function already printed the exact guidance,
#          1 = failure (a specific warning was already printed),
#          2 = no changes needed, or changes were written to startup files,
#              but the current shell still must be reloaded or sourced before
#              dcode will resolve,
#          3 = root install to a custom bin; PATH changes are left to MDM policy,
#          4 = the user declined the prompt; nothing was written.
# rc=0, 3 and 4 suppress the caller's reload hint. Any path that edits a
# startup file must return 2 (or 1) — a written file never helps the shell that
# is running right now — and any path that writes nothing must not return 2,
# or the user is told to restart a shell for a change that was never made.
ensure_path_setup() {
  local binary_name="$1"
  local binary_path="$2"

  # Escape hatch for version-managed dotfiles (chezmoi, nix home-manager) and
  # MDM fleets: install and verify the binary, but never touch startup files
  # or create symlinks. The user adds the bin dir to PATH themselves.
  #
  # Accept the same *truthy* spellings as DEEPAGENTS_CODE_YES. The falsy
  # handling deliberately differs: ASSUME_YES maps anything unrecognized to
  # off, while this maps it to on. The opt-out exists to protect managed
  # dotfiles, so an unparsed value must fail toward leaving files alone rather
  # than editing them. Don't unify the two parsers.
  local no_modify_path
  no_modify_path="$(printf '%s' "${DEEPAGENTS_CODE_NO_MODIFY_PATH:-0}" \
    | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "$no_modify_path" in
    0|false|no|'') no_modify_path="0" ;;
    1|true|yes)    no_modify_path="1" ;;
    *)
      log_warn "Unrecognized DEEPAGENTS_CODE_NO_MODIFY_PATH value — treating it as 1 (leaving startup files alone)."
      no_modify_path="1"
      ;;
  esac
  if [ "$no_modify_path" = "1" ]; then
    log_info "Skipping PATH setup (DEEPAGENTS_CODE_NO_MODIFY_PATH is set)."
    log_info "  ${binary_name} is at ${binary_path} — add its directory to PATH yourself, e.g.:"
    log_info "    export PATH=\"${binary_path%/*}:\$PATH\""
    # rc=0 so the caller stays quiet: the generic reload hint would name
    # ~/.local/bin, contradicting the exact directory printed just above.
    return 0
  fi

  # uv's env file only exposes binaries that are actually under ~/.local/bin.
  # A custom uv tool bin still needs a symlink or its own PATH entry.
  local binary_dir="${binary_path%/*}"
  if [ -f "$HOME/.local/bin/env" ] && \
    paths_are_same_file "$binary_dir" "$HOME/.local/bin"; then
    return 2
  fi
  if [ "$(id -u)" -eq 0 ] && ! paths_are_same_file "$binary_dir" "$HOME/.local/bin"; then
    log_warn "${binary_name} installed to ${TOOL_BIN_DIR_DISPLAY}, which is not on the target user's PATH."
    log_warn "  Add that directory through the user's shell configuration or MDM policy."
    return 3
  fi

  # Step 1: try symlinking into a dir already in PATH (no profile change).
  if try_symlink_in_path "$binary_name" "$binary_path"; then
    if [ "$VERBOSE" = "1" ]; then
      log_success "Created symlink in PATH for ${binary_name}."
    fi
    return 0
  fi

  # Step 2: create ~/.local/bin, symlink there, then add to shell profiles.
  local local_bin_preexisted=false
  if [ -d "$HOME/.local/bin" ]; then
    local_bin_preexisted=true
  fi
  mkdir -p "$HOME/.local/bin" 2>/dev/null || {
    log_warn "Could not create ~/.local/bin."
    return 1
  }
  if [ "$local_bin_preexisted" = false ]; then
    fix_file_owner "$HOME/.local/bin"
  fi
  local symlink_path="$HOME/.local/bin/$binary_name"
  if ! paths_are_same_file "$binary_path" "$symlink_path"; then
    if [ -e "$symlink_path" ] && [ ! -L "$symlink_path" ]; then
      log_warn "Refusing to replace existing file at ${symlink_path}."
      return 1
    fi
    if [ -L "$symlink_path" ]; then
      rm -f "$symlink_path"
    fi
    if ! ln -s "$binary_path" "$symlink_path" 2>/dev/null; then
      log_warn "Could not create symlink at ${symlink_path}."
      return 1
    fi
    fix_file_owner "$symlink_path"
  fi

  # Step 3: detect the current shell, then write the PATH entry into every
  # relevant startup file that exists — plus the primary file for the current
  # shell — so dcode resolves no matter which shell the user opens next.
  detect_shell_profile
  # Reuse detect_shell_profile's result rather than re-deriving it from $SHELL,
  # so the two can't disagree about which shell this is: any second derivation
  # needs its own fallback for an empty $SHELL, and a mismatch silently drops
  # the primary profile detect_shell_profile just chose.
  local shell_name="$DETECTED_SHELL"
  local zdotdir
  zdotdir="$(resolve_zdotdir)"

  # Build the candidate profile list (deduped):
  #   - the current shell's primary file, even if it doesn't exist yet
  #     (zsh and bash only; fish is covered by its conf.d file below)
  #   - the zshrc at the resolved ZDOTDIR, plus ~/.zshrc when both exist
  #     (ZDOTDIR wins for the shell itself; ~/.zshrc covers stale copies)
  #   - whenever any bash file exists, or the shell is bash: .bashrc (on macOS
  #     only if it already exists), plus the login file (.bash_profile
  #     preferred, else .bash_login, else ~/.profile, which is what a bash
  #     login shell falls back to reading)
  #   - fish: conf.d/deepagents-code.fish, when fish is the shell or a fish
  #     config dir exists
  #   - any other shell: ~/.profile, in addition to whichever of the above
  #     matched on file existence
  local candidates=() seen=""
  add_candidate() {
    case ":$seen:" in
      *":$1:"*) ;;
      *)
        seen="$seen:$1"
        candidates+=("$1")
        ;;
    esac
  }
  has_candidate() {
    case ":$seen:" in
      *":$1:"*) return 0 ;;
    esac
    return 1
  }
  case "$shell_name" in
    zsh)
      # detect_shell_profile resolves ZDOTDIR too, so SHELL_PROFILE is already
      # the same "${zdotdir:-$HOME}/.zshrc" the zsh branch below adds, and
      # add_candidate dedupes. Adding it here is therefore redundant either
      # way; the guard just makes the single source of that path obvious.
      if [ -z "$zdotdir" ] || [ "$zdotdir" = "$HOME" ]; then
        [ -n "$SHELL_PROFILE" ] && add_candidate "$SHELL_PROFILE"
      fi
      ;;
    bash)
      [ -n "$SHELL_PROFILE" ] && add_candidate "$SHELL_PROFILE"
      ;;
    # fish is covered solely by its conf.d file below — never edit config.fish.
  esac
  if [ "$shell_name" = "zsh" ] || [ -f "${zdotdir:-$HOME}/.zshrc" ]; then
    add_candidate "${zdotdir:-$HOME}/.zshrc"
    # ~/.zshrc is added alongside a ZDOTDIR zshrc only when it already exists.
    # When ZDOTDIR is already set in the *environment*, zsh reads
    # ${ZDOTDIR}/.zshenv rather than ~/.zshenv, so zsh's live ZDOTDIR can
    # differ from whatever resolve_zdotdir parsed out of ~/.zshenv — and on a
    # machine set up that way, ~/.zshrc may not exist at all.
    if [ -n "$zdotdir" ] && [ "$zdotdir" != "$HOME" ] && [ -f "$HOME/.zshrc" ]; then
      add_candidate "$HOME/.zshrc"
    fi
  fi
  if [ "$shell_name" = "bash" ] || [ -f "$HOME/.bashrc" ] || \
    [ -f "$HOME/.bash_profile" ] || [ -f "$HOME/.bash_login" ]; then
    # .bashrc is sourced for interactive non-login shells (the default on
    # Linux); macOS Terminal runs login shells instead, so a .bashrc created
    # here wouldn't be read by the default Terminal session — skip creating
    # one there. (A non-login bash started by hand on macOS does read it,
    # which is why an existing .bashrc is still updated.)
    if [ -f "$HOME/.bashrc" ] || [ "$OS" != "macos" ]; then
      add_candidate "$HOME/.bashrc"
    fi
    # Test existence *or* already-queued: the bash branch above may have queued
    # a not-yet-created ~/.bash_profile as the shell's primary file. Looking
    # only at the filesystem would miss that, fall through to ~/.profile, and
    # create both — after which bash reads the .bash_profile we just made and
    # never the .profile, leaving a stray file behind.
    if [ -f "$HOME/.bash_profile" ] || has_candidate "$HOME/.bash_profile"; then
      add_candidate "$HOME/.bash_profile"
    elif [ -f "$HOME/.bash_login" ] || has_candidate "$HOME/.bash_login"; then
      add_candidate "$HOME/.bash_login"
    else
      # bash files exist but no login file. A bash login shell reads the first
      # of .bash_profile / .bash_login / .profile that exists, so with neither
      # of the first two present, ~/.profile is the file it will read — and
      # ~/.bashrc alone leaves login shells (macOS Terminal, `bash -l`, ssh)
      # without the entry. add_candidate dedupes if it's already listed.
      add_candidate "$HOME/.profile"
    fi
  fi
  if [ "$shell_name" = "fish" ] || [ -d "${XDG_CONFIG_HOME:-$HOME/.config}/fish" ]; then
    add_candidate "$(fish_conf_file)"
  fi
  case "$shell_name" in
    zsh|bash|fish) ;;
    *) add_candidate "$HOME/.profile" ;;
  esac

  # Defensive only: every branch above adds at least one candidate (unknown
  # shells fall through to ~/.profile), so this should be unreachable.
  if [ ${#candidates[@]} -eq 0 ]; then
    log_warn "${binary_name} installed to ~/.local/bin but no startup file could be selected."
    log_warn "  Add ~/.local/bin to your PATH manually."
    return 1
  fi

  # Collapse $HOME prefix to ~ for tidier display paths. Done with a case +
  # prefix strip rather than "${1/#$HOME/\~}": bash 3.2 (still the default
  # /bin/bash on macOS, and this script is run as `curl … | bash`) does not
  # unescape the \~ in a pattern-substitution replacement, so that form
  # renders literally as "\~/.zshrc".
  tilde_display() {
    case "$1" in
      "$HOME"/*) printf '~%s\n' "${1#"$HOME"}" ;;
      "$HOME")   printf '~\n' ;;
      *)         printf '%s\n' "$1" ;;
    esac
  }

  # A candidate is satisfied when it already has our managed block with the
  # right line, or (legacy installs, manual entries) any ~/.local/bin PATH
  # reference at all.
  profile_satisfied() {
    local profile="$1" export_line="$2"
    if managed_path_block_present "$profile"; then
      managed_path_block_has_line "$profile" "$export_line"
      return $?
    fi
    local_bin_in_profile "$profile"
  }

  # The line to write is a property of the *target file*, never of the current
  # shell. Using PATH_EXPORT here would write fish syntax (`fish_add_path …`)
  # into a fish user's ~/.zshrc — which is a candidate whenever that file
  # exists — so every zsh session would fail with `command not found:
  # fish_add_path` and never get ~/.local/bin. Only the fish conf.d file gets
  # fish syntax; everything else (.zshrc, .bashrc, .bash_profile, .profile) is
  # POSIX-compatible.
  export_line_for() {
    if [ "$1" = "$(fish_conf_file)" ]; then
      printf '%s' "$FISH_PATH_EXPORT"
    else
      printf '%s' "$POSIX_PATH_EXPORT"
    fi
  }

  # Partition candidates into satisfied vs needing the entry.
  local pending=() satisfied=()
  local profile export_line
  for profile in "${candidates[@]}"; do
    export_line="$(export_line_for "$profile")"
    if profile_satisfied "$profile" "$export_line"; then
      satisfied+=("$profile")
    else
      pending+=("$profile")
    fi
  done

  if [ ${#pending[@]} -eq 0 ]; then
    if [ "$VERBOSE" = "1" ]; then
      for profile in "${satisfied[@]}"; do
        log_info "~/.local/bin already configured in $(tilde_display "$profile")."
      done
    fi
    # Nothing to write, but the current shell may still lack ~/.local/bin on
    # PATH (stale shell). Return 2 so the caller emits a reload/source hint.
    return 2
  fi

  # Prompt interactively (once, covering all files), or auto-add when
  # non-interactive.
  # Name every file the answer covers. Consent for one file must not be taken
  # as consent to edit three: a macOS zsh user with a leftover ~/.bashrc and
  # ~/.bash_profile would otherwise be asked only about ~/.zshrc.
  local should_add=true prompt_targets
  prompt_targets="$(tilde_display "${pending[0]}")"
  # `pending` is guaranteed non-empty above, so "${pending[@]:1}" is safe even
  # on the bash 3.2 macOS still ships (an empty *slice* of a non-empty array is
  # fine there; only expanding a wholly empty array trips `set -u`). The length
  # check just skips a pointless loop.
  if [ ${#pending[@]} -gt 1 ]; then
    local extra
    for extra in "${pending[@]:1}"; do
      prompt_targets="${prompt_targets}, $(tilde_display "$extra")"
    done
  fi
  if [ "$IS_INTERACTIVE" = true ] && can_prompt; then
    if ! prompt_yn "Add ~/.local/bin to your PATH in ${prompt_targets}?"; then
      should_add=false
    fi
  fi
  if [ "$should_add" = false ]; then
    log_info "Skipped modifying shell startup files."
    # PATH_EXPORT is empty for unknown shells; fall back to the POSIX line so
    # this never renders as a bare "add to PATH:" with nothing after it.
    log_info "  To use ${binary_name}, add to PATH:  ${PATH_EXPORT:-$POSIX_PATH_EXPORT}"
    # rc=4, not 2: nothing was written, so the caller's "Restart your shell"
    # hint would be actively wrong — a restart cannot pick up an edit that was
    # never made. The exact line to run was just printed above.
    return 4
  fi

  local updated=0 failed=0
  for profile in "${pending[@]}"; do
    export_line="$(export_line_for "$profile")"
    # Create the file (and parents) if missing. Keep the real error text —
    # "Permission denied", "Read-only file system" and "Not a directory" call
    # for completely different fixes, and this warning is the user's only
    # signal that a startup file was skipped.
    if [ ! -f "$profile" ]; then
      local mk_err=""
      if ! mk_err=$(mkdir -p "$(dirname "$profile")" 2>&1); then
        log_warn "Could not create the directory for $(tilde_display "$profile"): ${mk_err}"
        failed=$((failed + 1))
        continue
      fi
      if ! mk_err=$(touch "$profile" 2>&1); then
        log_warn "Could not create $(tilde_display "$profile"): ${mk_err}"
        failed=$((failed + 1))
        continue
      fi
    fi
    # Check for an existing managed block first, including for fish: a fish
    # conf.d file that already has our block (with a stale line) must be
    # rewritten, not appended to, or every export-line change leaves another
    # dead block behind. The standalone-file argument only means our block
    # can't collide with *user* config — it can still collide with our own.
    if managed_path_block_present "$profile"; then
      # Our block exists with a stale line — rewrite it in place.
      if rewrite_managed_path_block "$profile" "$export_line"; then
        fix_file_owner "$profile"
        log_success "Updated deepagents-code PATH block in $(tilde_display "$profile")."
        updated=$((updated + 1))
      else
        log_warn "Could not update deepagents-code PATH block in $(tilde_display "$profile")."
        # Deliberately hedged: rewrite_managed_path_block also returns 1 for an
        # unresolvable symlink, a mktemp failure and a failed mv, each of which
        # already printed its own reason just above. Asserting the marker cause
        # outright would contradict those.
        log_warn "  One cause is a stray '# >>> deepagents-code installer >>>' line with no matching '<<<' line."
        failed=$((failed + 1))
      fi
    elif append_managed_path_block "$profile" "$export_line"; then
      fix_file_owner "$profile"
      log_success "Added ~/.local/bin to PATH in $(tilde_display "$profile")."
      updated=$((updated + 1))
    else
      log_warn "Could not update $(tilde_display "$profile")."
      failed=$((failed + 1))
    fi
  done

  # Never return 0 here. rc=0 tells the caller "PATH is already fixed for the
  # running shell", which is only true for the symlink path above — a file we
  # just edited does nothing for the shell that is currently running, so the
  # user still needs the reload hint. A failure on any candidate returns 1
  # rather than being masked by a success elsewhere: the file that failed may
  # be the only one this user's shell actually reads.
  if [ "$failed" -gt 0 ]; then
    return 1
  fi
  return 2
}

classify_shadowing_command() {
  local path="$1"
  case "$path" in
    /opt/homebrew/*|/usr/local/*)
      if [ "$OS" = "macos" ]; then
        printf 'Homebrew-managed'
        return 0
      fi
      ;;
    *pipx*)
      printf 'pipx-managed'
      return 0
      ;;
    */.local/bin/*)
      printf 'user-local'
      return 0
      ;;
  esac
  printf 'existing'
}

detect_shadowing_install() {
  local candidate expected original manager
  for candidate in dcode deepagents-code; do
    expected="${TOOL_BIN_DIR}/${candidate}"
    [ -x "$expected" ] || continue
    original=$(PATH="$ORIGINAL_PATH" command -v "$candidate" 2>/dev/null || true)
    [ -n "$original" ] || continue
    # Same file reached via a different PATH spelling (e.g. ~/.local/share/../bin
    # vs ~/.local/bin) is not a shadowing install. Keep the string equality as a
    # fast path, but also accept `-ef` (true when both paths are the same device
    # and inode, resolving `..` and symlinks) so aliases don't trigger a false
    # "existing install" warning. A genuinely different binary has a distinct
    # inode and still fails both checks, so only the false positive is skipped.
    { [ "$original" = "$expected" ] || [ "$original" -ef "$expected" ]; } && continue
    manager=$(classify_shadowing_command "$original")
    log_warn "Detected ${manager} ${candidate} at ${original}."
    log_warn "PATH order may run that binary instead of the uv tool at ${expected}."
    log_warn "Restart your shell after this installer updates PATH, or remove the older install."
  done
}

DCODE_BIN=""
DCODE_NAME=""
# Tracks whether the binary would have resolved via the user's original PATH,
# not the installer-mutated PATH.
DCODE_ON_PATH=false
for candidate in dcode deepagents-code; do
  if [ -x "${TOOL_BIN_DIR}/${candidate}" ]; then
    DCODE_BIN="${TOOL_BIN_DIR}/${candidate}"
    DCODE_NAME="$candidate"
    original=$(PATH="$ORIGINAL_PATH" command -v "$candidate" 2>/dev/null || true)
    if [ -n "$original" ] && \
      { [ "$original" = "$DCODE_BIN" ] || [ "$original" -ef "$DCODE_BIN" ]; }; then
      DCODE_ON_PATH=true
    fi
    break
  fi
done
if [ -z "$DCODE_BIN" ]; then
  for candidate in dcode deepagents-code; do
    if resolved=$(command -v "$candidate" 2>/dev/null) && [ -n "$resolved" ]; then
      DCODE_BIN="$resolved"
      DCODE_NAME="$candidate"
      original=$(PATH="$ORIGINAL_PATH" command -v "$candidate" 2>/dev/null || true)
      if [ -n "$original" ] && \
        { [ "$original" = "$DCODE_BIN" ] || [ "$original" -ef "$DCODE_BIN" ]; }; then
        DCODE_ON_PATH=true
      fi
      break
    fi
  done
fi

# Collapse $HOME prefix to ~ for a tidier display path. Used in user-facing
# log lines only; DCODE_BIN keeps the absolute path for any exec needs.
DCODE_BIN_DISPLAY="$DCODE_BIN"
if [ -n "$DCODE_BIN" ] && [ -n "${HOME:-}" ]; then
  case "$DCODE_BIN" in
    "$HOME"/*) DCODE_BIN_DISPLAY="~${DCODE_BIN#"$HOME"}" ;;
  esac
fi

detect_shadowing_install

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
  log_warn "dcode (or deepagents-code) command not found in ${TOOL_BIN_DIR_DISPLAY} or PATH. Restart your shell or run:"
  log_warn "  source ~/.zshrc   # (or ~/.bashrc)"
fi

# The binary verified via its absolute path but isn't on the current shell's
# PATH (typical right after a fresh `uv tool install`): typing `dcode` won't
# work until the shell picks up ~/.local/bin. Instead of just telling the user
# to restart their shell, try to fix the PATH now — symlink into an existing
# PATH dir, or add ~/.local/bin to the shell profile — so the binary is
# immediately usable in a new terminal without manual configuration.
if [ "$VERIFY_OK" = true ] && [ "$DCODE_ON_PATH" = false ] && [ -n "$DCODE_BIN" ]; then
  path_setup_rc=0
  ensure_path_setup "$DCODE_NAME" "$DCODE_BIN" || path_setup_rc=$?
  if [ "$path_setup_rc" -ne 0 ] && [ "$path_setup_rc" -ne 3 ] && \
    [ "$path_setup_rc" -ne 4 ]; then
    # rc=1: ensure_path_setup printed a specific warning; add the fallback.
    # rc=2: startup files were written, or no change was needed — either way
    #   the *running* shell still lacks ~/.local/bin on PATH, so emit the
    #   reload/source hint.
    # rc=0/3/4 are silent here: PATH is already fixed, MDM owns it, or the user
    #   declined and was already given the one line that would help.
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
      if ripgrep_setup_out=$(mktemp 2>/dev/null); then
        register_temp "$ripgrep_setup_out"
        if "$DCODE_BIN" tools install >"$ripgrep_setup_out" 2>&1; then
          fix_owner "${HOME}/.deepagents/bin"
        else
          echo ""
          cat "$ripgrep_setup_out" >&2 2>/dev/null || true
          ripgrep_managed_failed
        fi
        rm -f "$ripgrep_setup_out"
      else
        log_warn "Could not create a secure temp file; skipping managed ripgrep setup."
        ripgrep_managed_failed
      fi
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
