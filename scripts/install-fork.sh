#!/usr/bin/env bash
# Swap the installed `fastedits` CLI for this fork, or revert to upstream.
#
# This fork is deliberately NOT published to PyPI (PyPI stays upstream's
# release channel). This script IS the fork's distribution mechanism:
# it points `uv tool install` at the fork's git repo instead.
#
# The fork and upstream share the same PyPI name (`fastedits`) and the
# same console-script names (fastedit, fastedit-hook, fastedit-mcp), so
# a leftover install from ANY method collides with the one this script
# just installed, and whichever sits first on PATH silently wins. That is
# why uninstall sweeps every method (uv tool / pipx / pip) and the
# postflight step below resolves the binary that will actually run and
# confirms it is the fork, not just that *a* fastedit exists somewhere.
set -euo pipefail

FORK_URL="https://github.com/Emasoft/fastedit"
PACKAGE="fastedits"
REF="feat/create-file"
EXTRAS=""
# Whether --extras was PASSED, tracked separately from its value. `--extras ""`
# is a deliberate opt-out and must be distinguishable from never passing the
# flag at all; testing `-z "$EXTRAS"` alone conflates the two and silently
# installs a backend the user explicitly declined (measured: it did exactly
# that before this flag existed).
EXTRAS_SET=0
REVERT=0
DRY_RUN=0
NO_MODEL=0

# Tolerable "nothing to do" outcomes across uv / pipx / pip. A real
# failure (permissions, corrupt env, network) must NOT match this and
# must still abort the script.
TOLERABLE_UNINSTALL_RE='not installed|nothing to uninstall|externally-managed-environment'

usage() {
  cat <<'EOF'
Usage: install-fork.sh [--ref REF] [--extras LIST] [--revert] [--no-model] [--dry-run]

  --ref REF      Branch, tag or commit SHA of the fork to install
                 (default: feat/create-file)
  --extras LIST  Comma-separated extras, e.g. mlx,mcp (default: none)
  --revert       Uninstall the fork and reinstall upstream fastedits
                 from PyPI (undoes the swap; leaves downloaded model
                 weights in place — they're shared with the fork)
  --no-model     Skip downloading the merge model (~3 GB). Ignored with
                 --revert, which never touches the model.
  --dry-run      Print the commands that would run; execute nothing
  -h, --help     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      REF="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      EXTRAS_SET=1
      shift 2
      ;;
    --revert)
      REVERT=1
      shift
      ;;
    --no-model)
      NO_MODEL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$EXTRAS" ]]; then
  PKG_SPEC="${PACKAGE}[${EXTRAS}]"
else
  PKG_SPEC="${PACKAGE}"
fi

# PREFLIGHT — fail before touching anything if uv is missing, so we
# never half-uninstall the current tool.
if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not on PATH. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Render argv as a shell-quoted line a user can copy-paste and get the
# same invocation (so a spec like "fastedits @ git+https://..." stays one
# paste-able token). An argument made only of characters that never need
# quoting (word chars, path/URL punctuation) is left bare -- that's the
# common case (uv, tool, uninstall, fastedits) and unquoted reads best.
# Anything else is wrapped in single quotes -- far more readable than
# printf %q's backslash-per-space/bracket escaping -- except when the
# argument itself contains a single quote, where %q is used instead so
# correctness never depends on the nicer-looking branch.
quote_argv() {
  local out="" arg quoted
  for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Za-z0-9_./:+@=-]+$ ]]; then
      quoted="$arg"
    elif [[ "$arg" == *"'"* ]]; then
      printf -v quoted '%q' "$arg"
    else
      quoted="'${arg}'"
    fi
    out+="${out:+ }${quoted}"
  done
  printf '%s' "$out"
}

# Run one uninstall command, tolerating "there was nothing to remove"
# in whatever shape that tool spells it, without masking a real failure.
run_uninstall_step() {
  local label="$1"
  shift
  echo "+ $(quote_argv "$@")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  local out status=0
  out=$("$@" 2>&1) || status=$?
  if [[ -n "$out" ]]; then
    echo "$out"
  fi
  if [[ "$status" -ne 0 ]] && ! grep -qiE "$TOLERABLE_UNINSTALL_RE" <<<"$out"; then
    echo "error: ${label} failed" >&2
    exit "$status"
  fi
}

# Sweep every install method that could be shadowing the one we're about
# to put in place. uv is always attempted (it's our own preflight
# dependency); pipx/pip only if actually present on this machine.
sweep_uninstall() {
  local pkg="$1"
  echo "Sweeping other install locations for ${pkg} (uv tool, pipx, pip)..."
  run_uninstall_step "uv tool uninstall" uv tool uninstall "$pkg"

  if command -v pipx >/dev/null 2>&1; then
    run_uninstall_step "pipx uninstall" pipx uninstall "$pkg"
  fi

  local pipbin=""
  if command -v pip3 >/dev/null 2>&1; then
    pipbin="pip3"
  elif command -v pip >/dev/null 2>&1; then
    pipbin="pip"
  fi
  if [[ -n "$pipbin" ]]; then
    run_uninstall_step "${pipbin} uninstall" "$pipbin" uninstall -y "$pkg"
  fi
}

do_install() {
  local spec="$1"
  echo "+ uv tool install $(quote_argv "$spec")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  uv tool install "$spec"
}

# Same install, but a failure is REPORTED rather than fatal. This exists for
# exactly one caller: the optional backend-extra attempt below. The sweep
# uninstalls before it installs, so any install step that can fail is a step
# that can leave this machine with NO fastedit at all — and with no fastedit
# there is no sanctioned way to edit source and repair it. A compiled extra
# (mlx) is the only genuinely failure-prone spec this script builds, so it is
# the one install that must never be allowed to abort the run.
do_install_optional() {
  local spec="$1"
  echo "+ uv tool install $(quote_argv "$spec")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  uv tool install "$spec" || return 1
}

# `fastedit pull` (get_model_path) already resolves env var -> local dir
# -> cache dir -> download, in that order, so it's a no-op when the
# model is already cached. We don't re-implement that lookup here —
# duplicating fastedit's own cache path would just be a second, driftable
# copy of the same logic. We only decide WHICH model this platform wants.
detect_model() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    echo "mlx-8bit"
    return 0
  fi
  if [[ "$os" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "bf16"
    return 0
  fi
  return 1
}

# The backend EXTRA that makes detect_model's choice loadable. Only the
# Darwin/arm64 -> mlx pair is asserted here, because it is the only one
# measured: on 2026-09-13 an extras-less install on this machine cached 1.7 GB
# at mlx-8bit and then died with `ModuleNotFoundError: No module named 'mlx'`
# on the first model-merge edit. The Linux/bf16 case is deliberately NOT
# mapped -- which runtime serves bf16 was never verified here, and guessing
# `vllm` would put one of the most install-hostile packages in the ecosystem
# on the failure path of a platform this repo cannot test.
detect_backend_extra() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    echo "mlx"
    return 0
  fi
  return 1
}

pull_model() {
  local model="$1"
  echo "+ fastedit pull --model $(quote_argv "$model")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  fastedit pull --model "$model"
}

# A failed postflight is only useful if it tells the user how to fix it,
# not just what's wrong -- otherwise it gets shrugged off as noise ("it
# always says that") and the safety check it guards stops being trusted.
# The usual fix is the one `uv tool install` itself hints at moments
# earlier: uv's own bin dir isn't on PATH yet.
print_path_hint() {
  local uv_bin=""
  uv_bin=$(uv tool dir --bin 2>/dev/null || true)
  if [[ -n "$uv_bin" ]]; then
    echo "hint: uv installs tools into ${uv_bin} -- make sure it's on PATH." >&2
  fi
  echo "hint: run 'uv tool update-shell' (then open a new shell) or add uv's tool bin dir to PATH yourself." >&2
}

# POSTFLIGHT — the whole point of this script. Resolve the fastedit that
# will actually run (not just "one exists somewhere") and confirm it is
# the fork. A stale upstream earlier on PATH, or an upstream reinstalled
# by something else, must fail loudly here rather than pass silently.
verify_fork_install() {
  local resolved
  if ! resolved=$(command -v fastedit); then
    echo "error: no 'fastedit' found on PATH after install" >&2
    print_path_hint
    exit 1
  fi

  local all_matches
  all_matches=$(type -a fastedit 2>/dev/null | grep '^fastedit is ' | sed 's/^fastedit is //' || true)
  local match_count
  match_count=$(printf '%s\n' "$all_matches" | grep -c . || true)

  local help_output
  if ! help_output=$(fastedit --help 2>&1); then
    echo "error: 'fastedit --help' failed after install" >&2
    echo "$help_output" >&2
    exit 1
  fi

  local missing=()
  local cmd
  for cmd in create duplicate split join; do
    if ! grep -qw "$cmd" <<<"$help_output"; then
      missing+=("$cmd")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "error: the 'fastedit' on PATH (${resolved}) is missing fork subcommands: ${missing[*]}" >&2
    if [[ "$match_count" -gt 1 ]]; then
      echo "error: multiple 'fastedit' binaries are on PATH — a shadowed upstream may be winning:" >&2
      printf '  %s\n' "$all_matches" >&2
    fi
    echo "error: the install did not actually put the fork in place — aborting" >&2
    print_path_hint
    exit 1
  fi

  if [[ "$match_count" -gt 1 ]]; then
    echo "warning: multiple 'fastedit' binaries found on PATH; the one that wins is ${resolved}:" >&2
    printf '  %s\n' "$all_matches" >&2
  fi
  echo "verified: fastedit at ${resolved} lists create, duplicate, split, join"
}

if [[ "$REVERT" -eq 1 ]]; then
  echo "Reverting to upstream ${PACKAGE} from PyPI..."
  sweep_uninstall "$PACKAGE"
  do_install "$PKG_SPEC"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "reverted: ${PACKAGE} is now upstream (no create/duplicate/split/join — that's expected)"
  fi
else
  echo "Installing fork ${FORK_URL}@${REF} in place of upstream ${PACKAGE}..."
  sweep_uninstall "$PACKAGE"

  # Two-phase, and the order is the whole point. The model is only useful if
  # the backend that loads it is installed, so try the backend extra FIRST --
  # but never let that attempt decide whether this machine ends up with a
  # fastedit. `mlx` is a compiled wheel; resolution or build CAN fail, and the
  # sweep above has already removed the previous install. So a failure falls
  # back to the bare spec (the near-unfailable pure-Python one) and downgrades
  # to deterministic-edit-only, which still works.
  # BACKEND_READY is tracked rather than probed. It records what we INSTALLED,
  # which is not the same as what is IMPORTABLE: a wheel can install and still
  # fail to load (wrong arch under Rosetta, a broken build). So this closes the
  # common case, not the whole class -- if the import fails anyway, the pull is
  # a wasted download, not a corruption.
  #
  # 0 means "we have positive evidence there is no backend", NOT "we don't
  # know". Absence of a mapping is not evidence of an absent backend: on a
  # platform where we never attempted an install we must NOT infer failure,
  # because the user may already have a working runtime we know nothing about.
  # Downgrading their previously-working model pull on a guess would be the
  # same unverified inference we refused to encode in detect_backend_extra.
  # Tri-state, deliberately. BACKEND_ATTEMPTED distinguishes "we tried to
  # install a backend and it failed" from "we never tried, so we know
  # nothing" -- and ONLY the first is grounds for withholding the model.
  BACKEND_ATTEMPTED=0
  BACKEND_READY=0
  BACKEND_EXTRA=""
  if [[ "$EXTRAS_SET" -eq 0 ]] && BACKEND_EXTRA=$(detect_backend_extra); then
    BACKEND_ATTEMPTED=1
    if do_install_optional "${PACKAGE}[${BACKEND_EXTRA}] @ git+${FORK_URL}@${REF}"; then
      BACKEND_READY=1
    else
      echo "warning: installing the '${BACKEND_EXTRA}' backend extra failed; falling back to a bare install." >&2
      echo "warning: deterministic edits will work; model-merge edits will not, and the model pull is skipped." >&2
      echo "warning: to retry the backend later: uv tool install --force '${PACKAGE}[${BACKEND_EXTRA}] @ git+${FORK_URL}@${REF}'" >&2
      do_install "${PKG_SPEC} @ git+${FORK_URL}@${REF}"
    fi
  else
    # An explicit --extras is the user's call: honour it exactly and make no
    # backend guess on top of it.
    do_install "${PKG_SPEC} @ git+${FORK_URL}@${REF}"
    case ",${EXTRAS}," in *,mlx,*|*,vllm,*) BACKEND_READY=1 ;; esac
    # `--extras ""` is not ignorance, it is an instruction. The user just told
    # us not to install a backend, so this install provides none -- that is
    # POSITIVE evidence, and pulling 1.7 GB of weights for the backend they
    # declined would be the same "act on an inference" error as the Linux
    # branch above, with the sign flipped. An unmapped PLATFORM teaches us
    # nothing (attempted stays 0); an explicit empty --extras teaches us
    # something (attempted is 1). They are different states and must not
    # collapse into one flag value.
    if [[ "$EXTRAS_SET" -eq 1 && "$BACKEND_READY" -eq 0 ]]; then
      BACKEND_ATTEMPTED=1
    fi
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    verify_fork_install
  fi

  if [[ "$NO_MODEL" -eq 0 ]]; then
    if [[ "$BACKEND_ATTEMPTED" -eq 1 && "$BACKEND_READY" -eq 0 ]]; then
      # Skip ONLY on positive evidence: we tried to install the backend on this
      # platform and the install failed. Pulling ~1.7 GB that nothing can load
      # is what this script used to do, and it produced a green install whose
      # first model-merge edit raised ModuleNotFoundError.
      #
      # The condition is deliberately NOT `BACKEND_READY -eq 0` alone. That
      # would also catch the case where we never attempted an install (no
      # extra is mapped for this platform), and withhold the model from a
      # machine that may well have a working runtime already -- a downgrade of
      # previously-working behaviour, inferred from our own ignorance.
      # Two different reasons land here and they must not share a message: a
      # FAILED attempt (BACKEND_EXTRA names what we tried) versus the user
      # DECLINING extras. Telling someone who passed --extras "" that an
      # install "failed" sends them debugging a failure that never happened.
      if [[ -n "$BACKEND_EXTRA" ]]; then
        echo "note: the '${BACKEND_EXTRA}' backend failed to install — skipping the model pull (~1.7 GB nothing could load)."
        echo "note: install a backend first, then run 'fastedit pull' — see the warnings above."
      else
        echo "note: no backend extra was installed (--extras \"\") — skipping the model pull (~1.7 GB nothing could load)."
        echo "note: if you already have a backend, run 'fastedit pull' yourself, or re-run without --extras."
      fi
    elif MODEL=$(detect_model); then
      pull_model "$MODEL"
      if [[ "$DRY_RUN" -eq 0 ]]; then
        fastedit doctor || true
      fi
    else
      echo "note: no known model for this platform ($(uname -s)/$(uname -m)) — skipping model pull."
      echo "note: run 'fastedit pull --model mlx-8bit' (Apple Silicon) or 'fastedit pull --model bf16' (Linux GPU) manually if needed."
    fi
  fi
fi
