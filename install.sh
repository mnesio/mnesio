#!/bin/sh
# mnesio installer — fetch a prebuilt `mnesio-code` + `mnesio-mcp`.
#
#   curl -fsSL https://raw.githubusercontent.com/mnesio/mnesio/main/install.sh | sh
#
# Why this exists: `mnesio-code` is only useful with 28 tree-sitter grammars
# compiled in, which from source is minutes of C compilation. That build now
# happens once in CI and this script downloads the result.
#
# POSIX sh, not bash — macOS ships bash 3.2 and some containers ship no bash at
# all. Nothing here needs arrays or [[.
#
# Environment:
#   MNESIO_VERSION  tag to install            (default: the latest release)
#   MNESIO_BIN_DIR  where to put the binaries (default: ~/.local/bin)
#
# It will not use sudo, will not edit your shell profile, and will not install
# over a different tool that happens to share the name. Those are all things a
# piped-to-shell script should ask about rather than assume.

set -eu

REPO="mnesio/mnesio"
BIN_DIR="${MNESIO_BIN_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"; }

need uname
need mkdir
need tar

# One of the two; curl is checked first because it is what the install line uses.
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
  fetch_stdout() { curl -fsSL "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
  fetch_stdout() { wget -qO- "$1"; }
else
  die "need curl or wget"
fi

# --- platform -----------------------------------------------------------
# Unsupported platforms stop here with the exact triple, rather than
# downloading a 404 page and handing over a "binary" that is HTML.
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Darwin) os_part="apple-darwin" ;;
  Linux)  os_part="unknown-linux-gnu" ;;
  MINGW*|MSYS*|CYGWIN*)
    die "Windows: download the .zip from https://github.com/$REPO/releases and add it to PATH" ;;
  *) die "unsupported OS: $os" ;;
esac
case "$arch" in
  arm64|aarch64) arch_part="aarch64" ;;
  x86_64|amd64)  arch_part="x86_64" ;;
  *) die "unsupported architecture: $arch" ;;
esac
target="$arch_part-$os_part"

# --- version ------------------------------------------------------------
version="${MNESIO_VERSION:-}"
if [ -z "$version" ]; then
  say "Resolving the latest release…"
  # The redirect from /releases/latest carries the tag, so this needs no JSON
  # parsing and no jq. `sed` extracts the trailing path segment.
  version="$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" \
    | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
  [ -n "$version" ] || die "could not determine the latest release — set MNESIO_VERSION=vX.Y.Z"
fi

name="mnesio-$version-$target"
url="https://github.com/$REPO/releases/download/$version/$name.tar.gz"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

say "Downloading $name…"
fetch "$url" "$tmp/$name.tar.gz" \
  || die "no build for $target at $version — see https://github.com/$REPO/releases"

# --- verify -------------------------------------------------------------
# A truncated download must fail here rather than land a corrupt binary on
# someone's PATH, where the symptom is an unexplained crash later.
if fetch "$url.sha256" "$tmp/sums" 2>/dev/null; then
  want="$(cut -d' ' -f1 < "$tmp/sums")"
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$tmp/$name.tar.gz" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    got="$(shasum -a 256 "$tmp/$name.tar.gz" | cut -d' ' -f1)"
  else
    got=""
    say "warning: no sha256 tool found — skipping checksum verification"
  fi
  if [ -n "$got" ] && [ "$got" != "$want" ]; then
    die "checksum mismatch — refusing to install
  expected $want
  got      $got"
  fi
else
  say "warning: no published checksum for this asset — cannot verify the download"
fi

tar xzf "$tmp/$name.tar.gz" -C "$tmp"

# --- install ------------------------------------------------------------
mkdir -p "$BIN_DIR" || die "cannot create $BIN_DIR"
[ -w "$BIN_DIR" ] || die "$BIN_DIR is not writable. Set MNESIO_BIN_DIR to somewhere you own."

for b in mnesio-code mnesio-mcp; do
  src="$tmp/$name/$b"
  [ -f "$src" ] || die "$b missing from the archive — the release is malformed"
  # Never silently replace an unrelated binary of the same name. Overwriting
  # our own is an upgrade; overwriting someone else's is a hijack.
  if [ -e "$BIN_DIR/$b" ] && ! "$BIN_DIR/$b" --help 2>&1 | grep -q mnesio; then
    die "$BIN_DIR/$b exists and is not mnesio — remove it or set MNESIO_BIN_DIR"
  fi
  install -m 755 "$src" "$BIN_DIR/$b" 2>/dev/null || {
    cp "$src" "$BIN_DIR/$b" && chmod 755 "$BIN_DIR/$b"
  }
done

say ""
say "Installed mnesio-code and mnesio-mcp $version to $BIN_DIR"

# Tell the truth about PATH rather than editing a shell profile behind someone's
# back — a piped-to-shell script has not earned that.
case ":$PATH:" in
  *":$BIN_DIR:"*) say "Run:  mnesio-code ." ;;
  *)
    say ""
    say "$BIN_DIR is not on your PATH. Add it:"
    say "    export PATH=\"\$PATH:$BIN_DIR\""
    say ""
    say "Or run it directly:  $BIN_DIR/mnesio-code ."
    ;;
esac
