#!/usr/bin/env bash
# Install deepagents-cli via uv.
#
# Interactive mode detection, color logging, and optional tool install
# patterns adapted from hermes-agent (NousResearch/hermes-agent).
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/cli/scripts/install.sh | bash
#
# Environment variables:
#   DEEPAGENTS_EXTRAS  — comma-separated pip extras, e.g. "anthropic",
#                        "anthropic,groq", or "daytona"
#                        (see pyproject.toml for available extras)
#   DEEPAGENTS_PYTHON  — Python version to use (default: 3.13)
#   DEEPAGENTS_SKIP_OPTIONAL — set to 1 to skip optional tool checks
#   UV_BIN             — path to uv binary (auto-detected if unset)
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
    log_error "For help, visit: https://docs.langchain.com/oss/python/deepagents/cli/overview"
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
EXTRAS="${DEEPAGENTS_EXTRAS:-}"
PYTHON_VERSION="${DEEPAGENTS_PYTHON:-3.13}"
SKIP_OPTIONAL="${DEEPAGENTS_SKIP_OPTIONAL:-0}"

# Validate and normalize extras: accept bare CSV, wrap in brackets for pip
if [[ -n "$EXTRAS" ]]; then
  # Strip brackets if the user passed them anyway
  EXTRAS="${EXTRAS#[}"
  EXTRAS="${EXTRAS%]}"
  if [[ ! "$EXTRAS" =~ ^[-a-zA-Z0-9,]+$ ]]; then
    log_error "DEEPAGENTS_EXTRAS must be comma-separated extra names, e.g. 'anthropic,groq' or 'daytona'"
    exit 1
  fi
  EXTRAS="[${EXTRAS}]"
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
# Install deepagents-cli
# ---------------------------------------------------------------------------
PACKAGE="deepagents-cli${EXTRAS}"

# Capture pre-install version (if any) for messaging
PRE_VERSION=""
if command -v deepagents >/dev/null 2>&1; then
  PRE_VERSION=$(deepagents -v 2>/dev/null | head -1 | awk '{print $NF}') || PRE_VERSION=""
elif [ -x "${HOME}/.local/bin/deepagents" ]; then
  PRE_VERSION=$("${HOME}/.local/bin/deepagents" -v 2>/dev/null | head -1 | awk '{print $NF}') || PRE_VERSION=""
fi

if [ -n "$PRE_VERSION" ]; then
  log_info "deepagents-cli ${PRE_VERSION} found — checking for updates..."
else
  log_info "Installing ${PACKAGE}..."
fi

if ! "$UV_BIN" tool install -U --python "$PYTHON_VERSION" "$PACKAGE"; then
  log_error "Failed to install ${PACKAGE}. See errors above."
  log_error "Common fixes: check your network, try a different Python version (DEEPAGENTS_PYTHON=3.12), or install manually."
  exit 1
fi
log_success "deepagents-cli installed."

# ---------------------------------------------------------------------------
# Post-install verification
# ---------------------------------------------------------------------------
DEEPAGENTS_BIN=""
if command -v deepagents >/dev/null 2>&1; then
  DEEPAGENTS_BIN="deepagents"
elif [ -x "${HOME}/.local/bin/deepagents" ]; then
  DEEPAGENTS_BIN="${HOME}/.local/bin/deepagents"
fi

if [ -n "$DEEPAGENTS_BIN" ]; then
  if VERSION=$("$DEEPAGENTS_BIN" -v 2>&1); then
    log_success "Verified: deepagents ${VERSION}"
  else
    log_warn "deepagents binary found but 'deepagents -v' failed:"
    log_warn "  ${VERSION}"
    log_warn "The installation may be broken. Try running: deepagents -v"
  fi
else
  log_warn "deepagents command not found in PATH. Restart your shell or run:"
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
  echo ""
  log_info "Checking optional tools..."

  if command -v rg >/dev/null 2>&1; then
    rg_version=$(rg --version 2>/dev/null | head -1 | awk '{print $2}') || rg_version="(version unknown)"
    log_success "ripgrep ${rg_version} found"
  else
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
# Done
# ---------------------------------------------------------------------------
echo ""
# shellcheck disable=SC2059
printf "${GREEN}✔${NC} Setup complete. Run: ${BOLD}deepagents${NC}\n"
echo ""
echo "For help and support, see the Deep Agents CLI docs:"
echo "  https://docs.langchain.com/oss/python/deepagents/cli/overview"
