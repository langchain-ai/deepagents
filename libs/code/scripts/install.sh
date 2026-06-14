#!/usr/bin/env bash
# Install deepagents-code.
#
# Usage:
#   curl -LsSf https://langch.in/dcode | bash
#
# Install an exact pre-release version:
#   curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_VERSION="0.1.0rc1" bash
#
# Allow uv to consider alpha/beta/rc releases when resolving the latest version:
#   curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_PRERELEASE="allow" bash
#
# DEEPAGENTS_CODE_VERSION and DEEPAGENTS_CODE_PRERELEASE are mutually exclusive:
# an exact pin already selects a single version, so setting both is an error.
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
#   DEEPAGENTS_CODE_EXTRAS  — comma-separated pip extras, e.g. "ollama",
#                             "ollama,groq", or "daytona"
#                             (see pyproject.toml for available extras)
#   DEEPAGENTS_CODE_VERSION — exact package version to install, e.g. "0.1.0rc1"
#                             (mutually exclusive with DEEPAGENTS_CODE_PRERELEASE)
#   DEEPAGENTS_CODE_PRERELEASE — uv pre-release strategy applied when resolving
#                                the latest version: disallow, allow,
#                                if-necessary, explicit, or
#                                if-necessary-or-explicit
#                                (mutually exclusive with DEEPAGENTS_CODE_VERSION)
#   DEEPAGENTS_CODE_PYTHON  — Python version to use (default: 3.13)
#   DEEPAGENTS_CODE_SKIP_OPTIONAL — set to 1 to skip optional tool checks
#   DEEPAGENTS_CODE_VERBOSE — set to 1 to show uv's raw stderr (timing
#                             lines, unfiltered package diff) and the
#                             quiet-by-default status lines (optional-tool
#                             checks, post-install footer); useful when
#                             debugging
#   UV_BIN                  — path to uv binary (auto-detected if unset)
#
# Credits:
#   Interactive mode detection, color logging, and optional tool install
#   patterns adapted from hermes-agent (NousResearch/hermes-agent).

set -euo pipefail

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
# Exit trap — ensures the user always sees an actionable message on failure
# ---------------------------------------------------------------------------
cleanup() {
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "" >&2
    log_error "Installation failed (exit code ${exit_code}). See errors above."
    log_error "For help, visit: https://docs.langchain.com/deepagents-code"
  fi
}
trap cleanup EXIT

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXTRAS="${DEEPAGENTS_CODE_EXTRAS:-}"
VERSION="${DEEPAGENTS_CODE_VERSION:-}"
PRERELEASE="${DEEPAGENTS_CODE_PRERELEASE:-}"
PYTHON_VERSION="${DEEPAGENTS_CODE_PYTHON:-3.13}"
SKIP_OPTIONAL="${DEEPAGENTS_CODE_SKIP_OPTIONAL:-0}"
VERBOSE="${DEEPAGENTS_CODE_VERBOSE:-0}"

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

# An exact pin already selects a single version, so a pre-release strategy
# (which only affects how a range resolves) is redundant at best and
# contradictory at worst (e.g. an rc pin with "disallow"). Reject the combo
# up front rather than forwarding an ambiguous request to uv.
if [[ -n "$VERSION" && -n "$PRERELEASE" ]]; then
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
install_uv() {
  if command -v curl >/dev/null 2>&1; then
    log_info "Downloading uv installer..."
    if ! curl -fsSL https://astral.sh/uv/install.sh | sh; then
      log_error "uv installation failed. See errors above."
      exit 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    log_info "Downloading uv installer..."
    if ! wget -qO- https://astral.sh/uv/install.sh | sh; then
      log_error "uv installation failed. See errors above."
      exit 1
    fi
  else
    log_error "curl or wget is required to install uv."
    exit 1
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  log_info "uv not found — installing..."
  install_uv
  fix_owner "${HOME}/.local/bin"  # root installs: restore user ownership
fi

# Resolve uv binary: honor UV_BIN override, then PATH, then the default
# install location (~/.local/bin). A fresh install may not have updated PATH
# in the current session, so we source the env file the installer creates.
if [ -z "${UV_BIN:-}" ]; then
  UV_BIN="uv"
  if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    if [ -f "${HOME}/.local/bin/env" ]; then
      # shellcheck source=/dev/null
      . "${HOME}/.local/bin/env"
    fi
  fi
  if ! command -v uv >/dev/null 2>&1; then
    UV_BIN="${HOME}/.local/bin/uv"
    if [ ! -x "$UV_BIN" ]; then
      log_error "uv not found after installation. Restart your shell or add ~/.local/bin to PATH."
      exit 1
    fi
  fi
fi

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
elif [ -n "$PRE_VERSION" ]; then
  log_info "deepagents-code ${PRE_VERSION} found — checking for updates..."
else
  log_info "Installing ${PACKAGE}..."
fi

# Capture uv stderr so we can:
#   1. Rewrite the cryptic "Ignoring existing environment …" warning into
#      plain English. uv emits that line when it rebuilds the tool venv
#      instead of upgrading in place (e.g., Python interpreter mismatch, or
#      editable↔regular install swap).
#   2. Drop uv's per-step timing lines ("Resolved N packages in...", etc.)
#      and the trailing "Installed N executables:" line — we already show
#      a Verified line with the binary name and version.
#   3. Reformat the `- pkg==X` / `+ pkg==Y` diff into an aligned
#      "pkg  X → Y" table under a single header.
# Using a tempfile (vs. process substitution) ensures we see uv's full exit
# status and don't race the warning past later log lines.
uv_stderr=$(mktemp 2>/dev/null) || uv_stderr="/tmp/deepagents-install.$$.err"
uv_rc=0
if [[ -n "$PRERELEASE" ]]; then
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
    /^Resolved [0-9]+ packages? in /    { next }
    /^Prepared [0-9]+ packages? in /    { next }
    /^Uninstalled [0-9]+ packages? in / { next }
    /^Installed [0-9]+ packages? in /   { next }
    /^Audited [0-9]+ packages? in /     { next }
    /^Checked [0-9]+ packages? in /     { next }
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
      maxw = 0
      any_removed = 0
      for (i = 1; i <= cnt; i++) {
        p = order[i]
        if (length(p) > maxw) maxw = length(p)
        if (p in removed) any_removed = 1
      }
      print (any_removed ? "Updated packages:" : "Installed packages:")
      for (i = 1; i <= cnt; i++) {
        p = order[i]
        pad = ""
        for (j = length(p); j < maxw; j++) pad = pad " "
        if ((p in removed) && (p in added)) {
          printf "  %s%s  %s → %s\n", p, pad, removed[p], added[p]
        } else if (p in added) {
          # "(new)" only disambiguates within an Updated list (mixed with
          # upgraded/removed rows). Under "Installed packages:" every row is
          # new, so the header already says it — drop the suffix.
          if (any_removed) printf "  %s%s  %s (new)\n", p, pad, added[p]
          else             printf "  %s%s  %s\n", p, pad, added[p]
        } else {
          printf "  %s%s  %s (removed)\n", p, pad, removed[p]
        }
      }
    }
  ' "$uv_stderr" >&2
else
  cat "$uv_stderr" >&2
fi
rm -f "$uv_stderr"
if [ "$uv_rc" -ne 0 ]; then
  log_error "Failed to install ${PACKAGE}. See errors above."
  log_error "Common fixes: check your network, try a different Python version (DEEPAGENTS_CODE_PYTHON=3.12), or install manually."
  exit 1
fi
fix_owner "${HOME}/.local/bin" "${HOME}/.local/share/uv"  # uv binaries + tool data
if [ "$OS" = "macos" ] && [ -d "${HOME}/Library/Caches/uv" ]; then
  fix_owner "${HOME}/Library/Caches/uv"
elif [ -d "${HOME}/.cache/uv" ]; then
  fix_owner "${HOME}/.cache/uv"
fi
# ---------------------------------------------------------------------------
# Post-install verification + contextual status
# ---------------------------------------------------------------------------
DCODE_BIN=""
DCODE_NAME=""
for candidate in dcode deepagents-code; do
  if resolved=$(command -v "$candidate" 2>/dev/null) && [ -n "$resolved" ]; then
    DCODE_BIN="$resolved"
    DCODE_NAME="$candidate"
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
  log_success "deepagents-code ${NEW_VERSION} already up to date."
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

if [ "$SKIP_OPTIONAL" != "1" ]; then
  if command -v rg >/dev/null 2>&1; then
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
# Done — footer wording depends on whether anything changed:
#   - already up to date  → "Already installed"
#   - fresh install / upgrade / editable→PyPI swap → "Setup complete"
# ---------------------------------------------------------------------------
if [ "$IS_EDITABLE" = false ] && [ -n "$PRE_VERSION" ] && [ -n "$NEW_VERSION" ] \
  && [ "$PRE_VERSION" = "$NEW_VERSION" ]; then
  footer_msg="Already installed."
else
  footer_msg="Setup complete."
fi
echo ""
# shellcheck disable=SC2059
printf "${GREEN}✔${NC} %s Run: ${BOLD}dcode${NC}\n" "$footer_msg"
echo "  Docs: https://docs.langchain.com/deepagents-code"
