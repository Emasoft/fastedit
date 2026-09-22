"""FastEdit CLI — edit, read, delete, move, rename, search, diff, undo.

Usage:
    fastedit read src/app.py
    fastedit edit src/app.py --snippet '...' --replace greet
    fastedit edit src/app.py --snippet - --after greet < snippet.py
    fastedit batch-edit src/app.py --edits '[{"snippet": "...", "replace": "fn"}]'
    fastedit delete src/app.py old_function
    fastedit move src/app.py my_func --after other_func
    fastedit rename src/app.py old_name new_name
    fastedit diff src/app.py
    fastedit undo src/app.py
    fastedit search "query" src/
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.resources
import sys
from pathlib import Path

from .cli_help import EPILOG
from .file_lock import FileLockedError, acquire_edit_lock

# ---------------------------------------------------------------------------
# Backend helpers (for model-using subcommands: edit, batch-edit, multi-edit)
# ---------------------------------------------------------------------------

def _make_backend_with_overrides(args):
    """Build the inference backend; CLI args override environment variables."""
    import os

    backend_kind = getattr(args, "backend", None) or os.environ.get("FASTEDIT_BACKEND", "mlx")

    if backend_kind == "vllm":
        from .inference.vllm_engine import VLLMEngine
        api_base = (
            getattr(args, "api_base", None)
            or os.environ.get("FASTEDIT_VLLM_API_BASE", "http://127.0.0.1:8000/v1")
        )
        model = (
            getattr(args, "api_model", None)
            or os.environ.get("FASTEDIT_VLLM_MODEL", "/root/fastedit-merged")
        )
        return backend_kind, VLLMEngine(
            api_base=api_base,
            model=model,
            api_key=os.environ.get("FASTEDIT_VLLM_API_KEY", "not-needed"),
            max_tokens=int(os.environ.get("FASTEDIT_VLLM_MAX_TOKENS", "16384")),
        )
    else:
        from .inference.mlx_engine import MLXEngine
        from .model_download import get_model_path
        model_path = getattr(args, "model_path", None) or get_model_path()
        return backend_kind, MLXEngine(model_path)


@contextlib.contextmanager
def _locked_for_edit(path: Path):
    """Hold the cross-process edit lock for *path* across this command's
    read→merge→write window.

    fastedit's writes are individually atomic, but two fastedit PROCESSES
    editing the same file interleave whole read-merge-write cycles and the
    second silently destroys the first's change. The lock is a kernel flock
    on a central lock file (~/.fastedit/locks/<sha256(realpath)>.lock), so
    the kernel releases it if a holder crashes — no stale locks, ever.
    Non-blocking: on conflict the command exits 1 naming the holder pid, how
    long it has held the lock, and the target path, before anything is read
    or written. ``--force`` deliberately does NOT bypass this — it is a
    parse/caller-gate opt-out, unrelated to concurrent instances. The B37
    expected-stat guard stays: it covers content changed by a NON-fastedit
    writer, a different failure than a lock held by fastedit itself.
    """
    try:
        with acquire_edit_lock(path):
            yield
    except FileLockedError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _format_small_file(path: str, content: str, total_lines: int) -> str:
    return f"{path} ({total_lines} lines — small file, showing full content)\n\n{content}"


def _format_structure(path: str, data: dict, total_lines: int) -> str:
    files = data.get("files", [])
    if not files:
        return f"{path} ({total_lines} lines) — no structure detected"

    file_info = files[0]
    language = data.get("language", "unknown")
    definitions = file_info.get("definitions", [])
    imports = file_info.get("imports", [])

    lines: list[str] = [f"{path} ({language}, {total_lines} lines)"]
    lines.append("")

    # Imports summary (deduplicated)
    if imports:
        seen: set[str] = set()
        import_names: list[str] = []
        for imp in imports:
            mod = imp.get("module", "")
            names = imp.get("names", [])
            label = f"{mod} ({', '.join(names)})" if names else mod
            if label not in seen:
                seen.add(label)
                import_names.append(label)
        lines.append(f"Imports: {', '.join(import_names)}")
        lines.append("")

    # Definitions with line ranges
    if definitions:
        class_ranges: list[tuple[int, int, str]] = []
        for d in definitions:
            if d.get("kind") == "class":
                class_ranges.append((d["line_start"], d["line_end"], d["name"]))

        for d in definitions:
            name = d.get("name", "?")
            kind = d.get("kind", "?")
            ls = d.get("line_start", 0)
            le = d.get("line_end", 0)
            sig = d.get("signature", "")

            indent = ""
            if kind == "method":
                for cs, ce, _cn in class_ranges:
                    if cs <= ls <= ce:
                        indent = "  "
                        break

            label = sig if sig else f"{kind} {name}"
            lines.append(f"{indent}L{ls}-{le:<4} {label}")

    return "\n".join(lines)


def cmd_read(args):
    """Show a file's structure (functions, classes, line ranges)."""
    import json as json_mod
    import subprocess

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = path.read_text(encoding="utf-8", errors="replace")
    total_lines = content.count("\n") + (
        1 if content and not content.endswith("\n") else 0
    )

    # Small files: return the content directly
    if total_lines <= 100:
        print(_format_small_file(args.file, content, total_lines))
        return

    try:
        result = subprocess.run(
            ["tldr", "structure", args.file, "--format", "compact"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            print(f"Error: tldr structure failed for {args.file}: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)

        data = json_mod.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Error: tldr structure timed out for {args.file}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        # tldr not found -- fall back to simple line count
        print(f"{args.file} ({total_lines} lines)")
        return
    except json_mod.JSONDecodeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(_format_structure(args.file, data, total_lines))

def _snippet_has_any_definition(snippet, ext="", first_original_line=None):
    """True if the snippet carries a definition line of its own.

    Deliberately name-AGNOSTIC: renaming through --replace is legitimate
    (the snippet defines a *different* name than the target), so requiring
    the targets own name here would refuse a working rename edit. This only
    asks "does the snippet define *something*", never "does it define the
    targets name".
    """
    import re

    from .inference.chunked_merge import (
        _DEFINITION_PATTERNS,
        _try_tldr_snippet_parse,
    )

    for line in snippet.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("//", "#", "*", "--")):
            continue
        for pattern in _DEFINITION_PATTERNS:
            if pattern.search(line):
                return True
        if re.search(
            r"\b(?:class|struct|enum|trait|interface|protocol|module|object|impl)\s+\w+",
            line,
        ):
            return True
    # The keyword-led floor above misses type-led C-family definitions
    # (``int target_fn(int a, int b) {``, ``public class Cache {``) and
    # similar declarations whose defining line carries no def/fn/func
    # keyword. Two generic, language-agnostic fallbacks before refusing a
    # --replace edit:
    #
    #   1. tldr parse — a snippet that parses as at least one top-level
    #      definition DOES carry its own definition line (covers renames
    #      of type-led definitions, where the definition line differs from
    #      the original's).
    #   2. definition-line restatement — a snippet whose first non-blank
    #      line restates the target's own first line (the definition line
    #      by AST construction) keeps the signature through the splice
    #      even when no parser recognizes the grammar's definitions.
    #
    # (Preserve-by-default Step 2: deterministic_edit now declines
    # single-line rewrites instead of replacing the gap, so this guard
    # decides whether the snippet is a complete replacement.) tldr missing
    # or misbehaving keeps the floor's answer.
    if first_original_line is not None:
        snippet_first = next(
            (ln.strip() for ln in snippet.splitlines() if ln.strip()), ""
        )
        if snippet_first and snippet_first == first_original_line.strip():
            return True
    if ext:
        return bool(_try_tldr_snippet_parse(snippet, ext))
    return False

def _snippet_is_single_matching_definition(snippet, target_node):
    """Python-only refinement: exactly one non-comment top-level node, of a
    type matching target_node.kind (a decorated def/class always counts,
    since the decorator wraps a real definition of either kind).
    """
    import textwrap

    from .data_gen.ast_analyzer import parse_code

    dedented = textwrap.dedent(snippet)
    tree = parse_code(dedented, "python")
    nodes = [c for c in tree.root_node.children if c.type != "comment"]
    if len(nodes) != 1:
        return False
    node = nodes[0]
    if node.type == "decorated_definition":
        return True
    kind_to_node_type = {
        "function": "function_definition",
        "method": "function_definition",
        "class": "class_definition",
    }
    return node.type == kind_to_node_type.get(target_node.kind)


def _try_deterministic_replace(path, original_code, original_lines, snippet, replace_sym, language, backups):
    from .data_gen.ast_analyzer import validate_parse
    from .inference.chunked_merge import (
        ChunkedMergeResult,
        _normalize_merged_eol,
        _qualified_symbol_names,
        _resolve_symbol,
        get_ast_map,
        get_ast_map_from_source,
    )
    from .inference.text_match import deterministic_edit, snippet_has_keep_marker
    from .split_join import (
        detect_line_ending,
        normalize_bare_cr_for_ast,
        normalize_line_endings,
    )

    total_lines = len(original_lines)
    # B3: parse the IN-MEMORY original first. get_ast_map consults the tldr
    # daemon, whose cache can hold pre-write line numbers for a file that
    # was just rewritten (B26 rationale) and which knows none of the B2/B3
    # data formats (html/json/yaml/...) — a replace= on those would refuse
    # "Symbol not found" even though the symbol map is available. The
    # same-length CR→LF substitution keeps the returned line numbers valid
    # against original_lines (the same protection delete_symbol uses). The
    # caller's detected `language` rides along as an explicit hint for
    # extension-unwired languages.
    ast_nodes = get_ast_map_from_source(
        normalize_bare_cr_for_ast(original_code), str(path), language,
    )
    if not ast_nodes:
        ast_nodes = get_ast_map(str(path), total_lines)
    target_node = _resolve_symbol(replace_sym, ast_nodes or [])
    if target_node is None:
        available = _qualified_symbol_names(ast_nodes or [])
        print(
            f"Error: Symbol '{replace_sym}' not found in {path}. "
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    func_start = target_node.line_start - 1  # 0-indexed
    func_end = target_node.line_end  # exclusive
    original_func = "".join(original_lines[func_start:func_end])

    # New/replaced content is normalized to the file's prevailing line
    # ending so the splice does not leave a mixed-ending seam; untouched
    # original_lines outside [func_start:func_end] are never touched.
    line_ending = detect_line_ending(original_code)

    # Try deterministic text-match first
    edited = deterministic_edit(original_func, snippet)
    if edited is not None:
        edited = normalize_line_endings(edited, line_ending)
        edited_lines = edited.splitlines(keepends=True)
        if edited_lines and not edited_lines[-1].endswith(("\n", "\r")):
            edited_lines[-1] += line_ending
        result_lines = list(original_lines)
        result_lines[func_start:func_end] = edited_lines
        merged = "".join(result_lines)
        # Step 14 (B31): the file's trailing-newline state comes from the
        # ORIGINAL via the central normalizer, not from the terminator this
        # branch appends for mid-file splices; the funnel also guarantees
        # the parse gate below sees the final bytes.
        merged = _normalize_merged_eol(merged, original_code)
        parse_valid = True
        if language:
            parse_valid = validate_parse(merged, language)
        if not parse_valid:
            # Same standard the direct-swap branch below applies to its own
            # output: a parse-invalid text-match splice (e.g. a brace-language
            # full-function snippet whose new body line cannot coexist with
            # the kept original body line) must not be handed back for the
            # parse gate to merely refuse. Fall through to chunked_merge,
            # whose replace= path declines parse-invalid splices and tries
            # the direct-swap / validated model route instead.
            return None
        return ChunkedMergeResult(
            merged_code=merged, parse_valid=parse_valid,
            chunks_used=0, chunk_regions=[], model_tokens=0, latency_ms=0.0,
        )

    # STOPGAP (TRDD-CMRMA2YG). "snippet IS the new symbol" is an assumption this
    # branch never checked. A snippet carrying a keep-marker is NOT a complete
    # symbol -- the marker stands in for lines the author deliberately did not
    # repeat -- so splicing it verbatim deletes exactly those lines. validate_parse
    # below CANNOT catch it: `#...` is a valid Python comment and `//...` a valid
    # JS one, so the mangled result parses clean and is written with exit 0.
    # The PROPER predicate is "the snippet parses as exactly one top-level symbol
    # named replace_sym"; that is filed separately. Refuse loudly meanwhile --
    # returning None would route to a merge backend that may not be installed and
    # surface as a bare ModuleNotFoundError, which tells the user nothing.
    if snippet_has_keep_marker(snippet):
        raise ValueError(
            f"snippet for '{replace_sym}' contains a keep-marker but no anchor line "
            f"matched the original body, so fastedit cannot place the edit. "
            f"Pass the full replacement body instead of a marker."
        )

    # DIRECT-SWAP PARSE GATE (exit-0 regression, Step 5 follow-up). Everything
    # below interprets the snippet as the WHOLESALE replacement of the target
    # symbol: the has_def floor above, the single-definition refinement, and the
    # direct line-range swap. That interpretation is only meaningful for a
    # snippet that PARSES. tree-sitter is error-tolerant — a snippet like
    # ``def f(:`` still yields a best-effort ``function_definition`` node — so
    # regex/AST *detection* of a definition line cannot tell broken from valid,
    # and the swap then produced parse-invalid output whose only remaining
    # handler was the model backend. The model "repairs" the syntax error and
    # the CLI reports success (exit 0) for code the user never wrote — a silent
    # rewrite of intent. Refuse loudly instead: a snippet that does not parse
    # has no edit semantics fastedit can honour, so nothing is written and the
    # caller fixes the snippet. Generic across languages (no per-grammar
    # branch); skipped when the file has no detected language, matching the
    # other parse gates in this function.
    if language and not validate_parse(snippet, language):
        raise ValueError(
            f"snippet for '{replace_sym}' is not valid {language} and cannot be "
            f"applied as a replacement. Fix the snippet's syntax and retry -- "
            f"fastedit will not guess at code that does not parse."
        )

    # TRDD-8M0MXRJO. A body-only snippet for a definition-kind target (function/
    # method/class/...) splices over the FULL AST line span -- signature line
    # included -- so a snippet missing its own definition line silently deletes
    # the targets signature. The result still parses (a lone `return 99` is
    # valid Python), so validate_parse below never catches it. Only refuse for
    # definition-kind targets (never a constant/variable, whose "signature" IS
    # its one line and is meant to be replaced whole), and only require SOME
    # definition line in the snippet -- not one named replace_sym, because a
    # --replace edit that legitimately renames the symbol defines a different
    # name on purpose.
    #
    # TRDD-CMRMA2YG follow-up. The line-based regex above cannot tell a real
    # top-level definition from a `def ...` that only appears as TEXT inside a
    # nested closure, a class body, or a multiline string literal -- each of
    # those still contains a matching line, so the guard passed while the
    # actual target symbol was destroyed. For python, tighten the check with
    # an AST-based single-definition test; other languages keep the
    # line-regex floor only (no AST refinement is defined for them).
    _DEFINITION_KINDS = {
        "function", "method", "class", "interface", "struct", "enum",
        "trait", "protocol", "module", "object", "impl",
    }
    has_def = _snippet_has_any_definition(
        snippet, ext=path.suffix, first_original_line=original_lines[func_start]
    )
    if has_def and language == "python":
        has_def = _snippet_is_single_matching_definition(snippet, target_node)
    if target_node.kind in _DEFINITION_KINDS and not has_def:
        raise ValueError(
            f"snippet for '{replace_sym}' (kind: {target_node.kind}) has no definition "
            f"line of its own, so splicing it over the symbol would delete its "
            f"signature. Pass the full replacement including the definition line, "
            f"or use --after to insert."
        )

    # Direct replacement: snippet IS the new symbol. Swap line ranges.
    snippet_text = normalize_line_endings(snippet, line_ending).rstrip("\r\n") + line_ending
    snippet_lines = snippet_text.splitlines(keepends=True)
    result_lines = list(original_lines)
    result_lines[func_start:func_end] = snippet_lines
    merged = "".join(result_lines)
    # Step 14 (B31): same funnel as the text-match branch above — whether
    # the edited file ends with a terminator is decided by the ORIGINAL's
    # state, never by the constant appended above for splicing.
    merged = _normalize_merged_eol(merged, original_code)
    parse_valid = True
    if language:
        parse_valid = validate_parse(merged, language)
    if parse_valid:
        return ChunkedMergeResult(
            merged_code=merged, parse_valid=True,
            chunks_used=0, chunk_regions=[], model_tokens=0, latency_ms=0.0,
        )
    # Parse invalid: fall through to model-based chunked_merge
    return None

def _refuse_if_edit_broke_parse(path, original_code, merged_code, language):
    """Raise when THIS edit is what broke the parse.

    Gated on the original parsing so a file that was already unparseable
    stays repairable by the only sanctioned writer on this machine; in
    that case we abstain and the caller's existing warning stands.
    """
    from .data_gen.ast_analyzer import validate_parse
    if not language:
        return
    if validate_parse(merged_code, language):
        return
    if not validate_parse(original_code, language):
        return
    raise ValueError(
        f"merged output for {path} has parse errors; refusing to write. "
        f"The file is unchanged."
    )

def cmd_edit(args):
    """Apply an edit snippet to a file using the FastEdit model."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    # Cross-process lock (both code paths below write this file): held from
    # here through the final write, released on every exit path.
    with _locked_for_edit(path):
        _cmd_edit_locked(args)


def _cmd_edit_locked(args):
    """cmd_edit's body, under the file's cross-process edit lock."""

    from .data_gen.ast_analyzer import detect_language
    from .inference.caller_safety import (
        _find_project_root,
        compute_signature_impact_note,
    )
    from .inference.chunked_merge import _validation_retries_metric, chunked_merge
    from .io_utils import UnsupportedEncodingError, read_source
    from .mcp.backup import (
        BackupStore,
        ConcurrentModificationError,
        _atomic_write,
    )

    snippet = sys.stdin.read() if args.snippet == "-" else args.snippet
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    backups = BackupStore()
    # B21: strict decode -- an undecodable byte must never become U+FFFD
    # and be written back as EF BF BD. The codec captured here (B23) flows
    # to every write below so untouched bytes round-trip exactly. B37: the
    # stat of that same open rides along so the writes refuse (instead of
    # clobbering) when the file changed on disk since this read.
    try:
        original_code, encoding, read_stat = read_source(path, return_stat=True)
    except UnsupportedEncodingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    language = detect_language(path)
    original_lines = original_code.splitlines(keepends=True)

    replace_sym = args.replace or None
    after_sym = args.after or None

    def _maybe_impact_note(merged_code: str) -> str:
        """Build the pre-flight impact note (VAL-M3-001) or empty str.

        Informational only -- the edit has already landed on disk by the
        time we call this. When replace= is not used or the signature
        is unchanged the helper returns None without invoking
        tldr (VAL-M3-002 hot path). We swallow any exception so
        infra hiccups cannot break the edit-success print.
        """
        if not replace_sym:
            return ""
        try:
            project_root = _find_project_root(path)
            note = compute_signature_impact_note(
                old_code=original_code,
                new_code=merged_code,
                symbol=replace_sym,
                language=language,
                file_path=path,
                project_root=project_root,
            )
        except Exception:  # noqa: BLE001 -- deliberate: we swallow any exception so an infra hiccup (tldr/AST) cannot fail the edit-success print (see docstring)
            return ""
        if not note:
            return ""
        return "\n" + note

    if replace_sym and not after_sym:
        try:
            result = _try_deterministic_replace(
                path, original_code, original_lines, snippet, replace_sym, language, backups,
            )
        except ValueError as e:
            # TRDD-CMRMA2YG: the keep-marker guard in _try_deterministic_replace
            # raises ValueError instead of returning None so the failure is a
            # clean diagnostic here, not a bare traceback nor a silent fall-through
            # to a merge backend that may not be installed.
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if result is not None:
            try:
                _refuse_if_edit_broke_parse(path, original_code, result.merged_code, language)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            try:
                _atomic_write(
                    path, result.merged_code, backups=backups, encoding=encoding,
                    expected_stat=read_stat,
                )
            except ConcurrentModificationError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            note = _maybe_impact_note(result.merged_code)
            print(
                f"Applied edit to {args.file}. "
                f"latency: 0ms, 0 tok/s, 0 tokens{note}"
            )
            return

    # Lazy backend: only loaded when merge_fn is actually called.
    # Deterministic paths (after=, replace= with text-match) never call merge_fn.
    _backend_cache = {}

    def _lazy_merge_fn(*a, **kw):
        if "engine" not in _backend_cache:
            _, engine = _make_backend_with_overrides(args)
            _backend_cache["engine"] = engine
        return _backend_cache["engine"].merge_auto(*a, **kw)

    try:
        result = chunked_merge(
            original_code=original_code,
            snippet=snippet,
            file_path=args.file,
            merge_fn=_lazy_merge_fn,
            language=language,
            after=after_sym,
            replace=replace_sym,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _refuse_if_edit_broke_parse(path, original_code, result.merged_code, language)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _atomic_write(
            path, result.merged_code, backups=backups, encoding=encoding,
            expected_stat=read_stat,
        )
    except ConcurrentModificationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    note = _maybe_impact_note(result.merged_code)

    tok_per_sec = (
        result.model_tokens / (result.latency_ms / 1000)
        if result.latency_ms > 0 else 0
    )
    metrics = (
        f"latency: {result.latency_ms:.0f}ms, "
        f"{tok_per_sec:.0f} tok/s, {result.model_tokens} tokens"
    )
    if result.chunks_used > 1:
        metrics += f", {result.chunks_used} chunk(s)"
    # Step A3: validation-retry count in the metrics segment (empty string
    # when none were consumed — shape stays stable). The retry budget itself
    # is resolved inside chunked_merge (FASTEDIT_MAX_RETRIES env or default).
    metrics += _validation_retries_metric(getattr(result, "retries", 0))
    if getattr(result, "chunks_rejected", 0):
        print(
            f"Warning: {result.chunks_rejected}/{result.chunks_used} chunk(s) rejected. "
            f"Partial edit applied. {metrics}"
        )
    elif language and not result.parse_valid:
        print(f"Warning: merged output has parse errors. Wrote anyway. {metrics}")

    print(f"Applied edit to {args.file}. {metrics}{note}")


def cmd_batch_edit(args):
    """Apply multiple sequential edits to one file."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    # Per-file scope (this command has exactly one target): the lock spans
    # the read→merge→write window, not "the whole batch" — but for a single
    # file those are the same window.
    with _locked_for_edit(path):
        _cmd_batch_edit_locked(args)


def _cmd_batch_edit_locked(args):
    """cmd_batch_edit's body, under the file's cross-process edit lock."""
    import json as json_mod

    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import (
        BatchEdit,
        _validation_retries_metric,
        batch_chunked_merge,
    )
    from .io_utils import UnsupportedEncodingError, read_source
    from .mcp.backup import (
        BackupStore,
        ConcurrentModificationError,
        _atomic_write,
    )

    edits_json = sys.stdin.read() if args.edits == "-" else args.edits
    try:
        edits_list = json_mod.loads(edits_json)
    except json_mod.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    batch = [
        BatchEdit(
            snippet=e["snippet"],
            after=e.get("after") or None,
            replace=e.get("replace") or None,
        )
        for e in edits_list
    ]

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    _backend_kind, backend = _make_backend_with_overrides(args)
    backups = BackupStore()
    # B21/B23: strict-decode read; the codec flows to the write below. B37:
    # the read-time stat guards the write against an external change.
    try:
        original_code, encoding, read_stat = read_source(path, return_stat=True)
    except UnsupportedEncodingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    language = detect_language(path)

    result = batch_chunked_merge(
        original_code=original_code,
        edits=batch,
        file_path=args.file,
        merge_fn=backend.merge_auto,
        language=language,
    )
    try:
        _refuse_if_edit_broke_parse(path, original_code, result.merged_code, language)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        _atomic_write(
            path, result.merged_code, backups=backups, encoding=encoding,
            expected_stat=read_stat,
        )
    except ConcurrentModificationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Applied {len(batch)} edits to {args.file}. "
        f"latency: {result.latency_ms:.0f}ms, {result.model_tokens} tokens"
        f"{_validation_retries_metric(getattr(result, 'retries', 0))}"
    )


def cmd_multi_edit(args):
    """Apply edits across multiple files, writing nothing unless every file succeeds."""
    import json as json_mod

    file_edits_json = sys.stdin.read() if args.file_edits == "-" else args.file_edits
    try:
        file_edits_list = json_mod.loads(file_edits_json)
    except json_mod.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Cross-process edit locks, per target: every existing target is locked
    # up front, non-blocking, in the caller's list order (a fixed order +
    # non-blocking acquisition cannot deadlock against another concurrent
    # multi-edit). Each lock is held from here through PHASE 3's write of
    # that target — that file's whole read→merge→write window, which is why
    # the locks must span the phases rather than sit inside one of them. A
    # conflict exits 1 before anything is read or written.
    with contextlib.ExitStack() as _target_locks:
        entries = file_edits_list if isinstance(file_edits_list, list) else []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("file_path"), str):
                target = Path(entry["file_path"])
                if target.exists():
                    _target_locks.enter_context(_locked_for_edit(target))
        _cmd_multi_edit_locked(args, file_edits_list)


def _cmd_multi_edit_locked(args, file_edits_list):
    """cmd_multi_edit's phases, under every target's cross-process edit lock."""
    import hashlib
    import os

    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import (
        BatchEdit,
        _validation_retries_metric,
        batch_chunked_merge,
    )
    from .io_utils import read_source
    from .mcp.backup import (
        BackupStore,
        ConcurrentModificationError,
        _atomic_write,
    )

    _backend_kind, backend = _make_backend_with_overrides(args)
    backups = BackupStore()

    # PHASE 1 -- reject EVERY knowable-beforehand error before touching anything.
    #
    # This used to validate, merge and write one file at a time in a single
    # loop, so a missing second target exited non-zero AFTER the first had been
    # written: the command reported failure and had modified the tree anyway
    # (TRDD-IUVBCTW3). That is worse than either a clean failure or a silent
    # success, because the error actively tells the caller nothing happened.
    #
    # Every problem is collected and reported, not just the first, so one run
    # tells the caller everything they must fix. Shape errors are checked here
    # rather than left to blow up as a KeyError mid-merge, and `is_file` is
    # checked rather than `exists` because a directory passes `exists` and then
    # raises IsADirectoryError during the merge phase -- both are knowable now.
    problems: list[str] = []
    for index, entry in enumerate(file_edits_list):
        if not isinstance(entry, dict) or "file_path" not in entry or "edits" not in entry:
            problems.append(f"entry {index}: expected an object with 'file_path' and 'edits'")
            continue
        path = Path(entry["file_path"])
        if not path.exists():
            problems.append(f"file not found: {path}")
        elif not path.is_file():
            problems.append(f"not a regular file: {path}")
        if not isinstance(entry["edits"], list) or not all(
            isinstance(e, dict) and "snippet" in e for e in entry["edits"]
        ):
            problems.append(
                f"entry {index}: 'edits' must be a list of objects each with 'snippet'"
            )
    if problems:
        for problem in problems:
            print(f"Error: {problem}", file=sys.stderr)
        sys.exit(1)

    # PHASE 2 -- compute every merge, still writing nothing. A merge that fails
    # here leaves the tree exactly as it was found. Each target's original bytes
    # are hashed as they are read, so PHASE 2.5 can detect a concurrent write
    # without keeping the (potentially large) original bytes around.
    #
    # The ValueError catch matters: batch_chunked_merge raises for a symbol that
    # does not exist and for a whole-file merge over the size limit. Letting
    # that escape would swap a corrupt tree for a bare traceback -- safe, but
    # telling the user nothing. The same swap was rejected elsewhere in this
    # codebase and is rejected here.
    pending: list[tuple[Path, str, int, object, str, os.stat_result, str]] = []
    for entry in file_edits_list:
        path = Path(entry["file_path"])
        batch = [
            BatchEdit(
                snippet=e["snippet"],
                after=e.get("after") or None,
                replace=e.get("replace") or None,
            )
            for e in entry["edits"]
        ]
        try:
            original_bytes = path.read_bytes()
            original_hash = hashlib.sha256(original_bytes).hexdigest()
            # B21/B23: strict-decode read (UnsupportedEncodingError is a
            # ValueError, caught below); the codec AND the read-time stat
            # (B37) ride with the pending entry so PHASE 3 writes with them.
            original_code, encoding, read_stat = read_source(path, return_stat=True)
            result = batch_chunked_merge(
                original_code=original_code,
                edits=batch,
                file_path=str(path),
                merge_fn=backend.merge_auto,
                language=detect_language(path),
            )
            _refuse_if_edit_broke_parse(path, original_code, result.merged_code, detect_language(path))
        except (ValueError, OSError) as e:
            print(f"Error: {path}: {e}", file=sys.stderr)
            print("Error: no files were modified.", file=sys.stderr)
            sys.exit(1)
        pending.append((path, result.merged_code, len(batch), result, original_hash, read_stat, encoding))

    # PHASE 2.5 -- re-verify every target BEFORE writing any of them.
    #
    # PHASE 2's merges can take seconds (they may call a model backend), so a
    # file can be modified -- or removed -- by something else between its own
    # read/hash and this point. Re-hashing target N immediately before writing
    # target N would be the wrong fix: by the time target N+1's re-hash caught
    # a change, target N would already be written -- reproducing the exact
    # TRDD-IUVBCTW3 partial-write defect this command exists to prevent, merely
    # re-triggered by a race instead of a merge error. Checking every target
    # here, before PHASE 3's write loop starts, is what makes "no target is
    # EVER written because of a race that was detectable" true.
    #
    # A target that vanished or became unreadable between PHASE 2 and here is
    # reported the same way as a target that changed -- it is unambiguously
    # "no longer what was read" -- rather than letting FileNotFoundError /
    # PermissionError escape as a bare traceback.
    changed: list[str] = []
    for path, _merged_code, _edit_count, _result, original_hash, _read_stat, _encoding in pending:
        try:
            still_matches = hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
        except OSError as e:
            changed.append(f"{path} (unreadable: {e})")
            continue
        if not still_matches:
            changed.append(str(path))
    if changed:
        for description in changed:
            print(
                f"Error: file changed since being read, refusing to write any target: {description}",
                file=sys.stderr,
            )
        print("Error: no files were modified.", file=sys.stderr)
        sys.exit(1)

    # PHASE 3 -- commit. Only reached when every target validated, every merge
    # succeeded, and PHASE 2.5 confirmed every target still matches what was
    # read.
    #
    # TWO HONEST LIMITS, both real and neither fixed here:
    #
    # 1. This is not a cross-file transaction. Each write is individually
    #    atomic, but a crash partway through this loop can still leave earlier
    #    files written. Real cross-file atomicity needs a journal.
    # 2. The read-to-write race is narrowed twice -- PHASE 2.5 re-verified
    #    every target before this loop, and each write below carries the
    #    stat captured at ITS read (B37): a target that changed on disk
    #    since its own read is REFUSED at write time, not overwritten from
    #    stale bytes. What remains is the residual window between the
    #    write-time stat and os.replace itself, and cross-file windows
    #    (refusing file N+2 cannot un-write file N). True cross-file
    #    atomicity needs a journal or file locks; neither exists here.
    #
    # What this DOES guarantee is that no file is written because of an error
    # that was knowable beforehand -- including a concurrent change to any
    # target that PHASE 2.5 or the write-time stat check could detect -- which
    # is the entire defect above.
    for path, merged_code, edit_count, result, _original_hash, read_stat, encoding in pending:
        try:
            _atomic_write(
                path, merged_code, backups=backups, encoding=encoding,
                expected_stat=read_stat,
            )
        except ConcurrentModificationError as e:
            print(f"Error: {path}: {e}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Applied {edit_count} edits to {path}. "
            f"latency: {result.latency_ms:.0f}ms, {result.model_tokens} tokens"
            f"{_validation_retries_metric(getattr(result, 'retries', 0))}"
        )


def cmd_delete(args):
    """Delete a function, method, or class from a file using AST analysis."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    with _locked_for_edit(path):
        _cmd_delete_locked(args)


def _cmd_delete_locked(args):
    """cmd_delete's body, under the file's cross-process edit lock."""
    from .data_gen.ast_analyzer import detect_language
    from .inference.caller_safety import (
        _find_project_root,
        check_cross_file_callers,
        format_refusal_message,
    )
    from .inference.chunked_merge import delete_symbol
    from .io_utils import UnsupportedEncodingError, read_source
    from .mcp.backup import (
        BackupStore,
        ConcurrentModificationError,
        _atomic_write,
    )

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    language = detect_language(path)
    # B21/B23: strict-decode read; the codec flows to the write below. B37:
    # the read-time stat guards the write against an external change.
    try:
        original_code, encoding, read_stat = read_source(path, return_stat=True)
    except UnsupportedEncodingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    backups = BackupStore()

    # Cross-file caller-safety check (M2). Skipped when --force is set.
    force = bool(getattr(args, "force", False))
    safety_note = ""
    if not force:
        project_root = _find_project_root(path)
        refs = check_cross_file_callers(
            file_path=path, symbol=args.symbol, project_root=project_root,
        )
        if refs:
            print(
                format_refusal_message(
                    args.symbol,
                    refs,
                    "Pass --force to delete anyway, or run "
                    "`fastedit rename-all` to migrate callers first.",
                ),
                file=sys.stderr,
            )
            sys.exit(2)

    try:
        result = delete_symbol(
            file_path=args.file,
            symbol=args.symbol,
            language=language,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _refuse_if_edit_broke_parse(path, original_code, result.merged_code, language)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _atomic_write(
            path, result.merged_code, backups=backups, encoding=encoding,
            expected_stat=read_stat,
        )
    except ConcurrentModificationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    warn = ""
    if language and not result.parse_valid:
        warn = " (warning: parse errors after delete)"
    print(
        f"Deleted {result.deleted_kind} '{result.deleted_symbol}' from {args.file}. "
        f"Removed L{result.deleted_lines[0]}-{result.deleted_lines[1]} "
        f"({result.lines_removed} lines). 0 model tokens.{warn}{safety_note}"
    )


def cmd_move(args):
    """Move a symbol to after another symbol in the same file."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    with _locked_for_edit(path):
        _cmd_move_locked(args)


def _cmd_move_locked(args):
    """cmd_move's body, under the file's cross-process edit lock."""
    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import move_symbol
    from .io_utils import UnsupportedEncodingError, read_source
    from .mcp.backup import (
        BackupStore,
        ConcurrentModificationError,
        _atomic_write,
    )

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    language = detect_language(path)
    # B21/B23: strict-decode read; the codec flows to the write below. B37:
    # the read-time stat guards the write against an external change.
    try:
        original_code, encoding, read_stat = read_source(path, return_stat=True)
    except UnsupportedEncodingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    backups = BackupStore()

    try:
        result = move_symbol(
            file_path=args.file,
            symbol=args.symbol,
            after=args.after,
            language=language,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _refuse_if_edit_broke_parse(path, original_code, result.merged_code, language)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _atomic_write(
            path, result.merged_code, backups=backups, encoding=encoding,
            expected_stat=read_stat,
        )
    except ConcurrentModificationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    warn = ""
    if language and not result.parse_valid:
        warn = " (warning: parse errors after move)"
    print(
        f"Moved {result.moved_kind} '{result.moved_symbol}' "
        f"from L{result.from_lines[0]}-{result.from_lines[1]} "
        f"to after '{result.after_symbol}' "
        f"(now L{result.new_lines[0]}-{result.new_lines[1]}) "
        f"in {args.file}. 0 model tokens.{warn}"
    )


def cmd_rename(args):
    """Rename all AST-verified references to a symbol in a single file.

    Drives matching through ``tldr references <name> <file> --scope file``,
    mirroring fast_rename_all's AST-verified behaviour. Substrings inside
    strings, comments, and docstrings are not renamed.
    """
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    with _locked_for_edit(path):
        _cmd_rename_locked(args)


def _cmd_rename_locked(args):
    """cmd_rename's body, under the file's cross-process edit lock."""
    import difflib

    from .inference.rename import do_rename_ast
    from .mcp.backup import BackupStore, _atomic_write

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    original = path.read_text(encoding="utf-8")

    renamed, count, skipped = do_rename_ast(path, args.old_name, args.new_name)

    if count == 0:
        print(
            f"Error: no code references to '{args.old_name}' found in {args.file} "
            f"(AST-verified via tldr references --scope file; matches inside "
            f"strings/comments/docstrings are not counted).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        skip_note = (
            f" (skipping {skipped} in strings/comments)" if skipped else ""
        )
        print(
            f"Dry run: would rename '{args.old_name}' -> '{args.new_name}' in "
            f"1 file, {count} replacement(s){skip_note}: {args.file}"
        )
        return

    backups = BackupStore()
    _atomic_write(path, renamed, backups=backups)

    skip_note = f" (skipped {skipped} in strings/comments)" if skipped else ""

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        renamed.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    print(
        f"Renamed '{args.old_name}' -> '{args.new_name}' in {args.file}: "
        f"{count} replacement(s).{skip_note} 0 model tokens."
    )
    print("".join(diff), end="")

def cmd_rename_all(args):
    """Rename a symbol across every supported file in a directory tree."""
    from .inference.rename import do_cross_file_rename
    from .mcp.backup import BackupStore, _atomic_write

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: directory not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    plan = do_cross_file_rename(
        root, args.old_name, args.new_name,
        kind_filter=getattr(args, "only", None),
    )
    if not plan:
        print(
            f"No occurrences of '{args.old_name}' found under {args.root} "
            f"(AST-verified via tldr — strings/comments/vendor dirs excluded).",
            file=sys.stderr,
        )
        sys.exit(1)

    total_count = sum(count for _, count, _ in plan.values())
    total_skipped = sum(skipped for _, _, skipped in plan.values())

    if args.dry_run:
        print(
            f"Dry run: would rename '{args.old_name}' -> '{args.new_name}' in "
            f"{len(plan)} file(s), {total_count} replacement(s)"
            + (f" (skipping {total_skipped} in strings/comments)" if total_skipped else "")
            + ":"
        )
        for path, (_, count, skipped) in sorted(plan.items()):
            skip_note = f" ({skipped} skipped)" if skipped else ""
            print(f"  {path} — {count} replacement(s){skip_note}")
        return

    backups = BackupStore()
    for path, (new_content, _, _) in plan.items():
        # Per-file scope: each target is locked only across its own write
        # (the reads happened in the plan above), so two concurrent
        # rename-all runs over overlapping trees serialize per file instead
        # of deadlocking on cross-file lock order.
        with _locked_for_edit(path):
            _atomic_write(path, new_content, backups=backups)

    skip_note = f" (skipped {total_skipped} in strings/comments)" if total_skipped else ""
    print(
        f"Renamed '{args.old_name}' -> '{args.new_name}' in {len(plan)} file(s), "
        f"{total_count} replacement(s).{skip_note} 0 model tokens."
    )

def cmd_move_to_file(args):
    """Move a symbol from one file to another and rewrite consumer imports.

    Delegates to :func:`fastedit.inference.move_to_file.move_to_file`. The
    helper handles AST extraction, destination insertion, and import
    rewrites in every importer discovered via ``tldr references``. On
    ``--dry-run`` nothing is written — we just print the plan.
    """
    from_path = Path(args.from_file)
    to_path = Path(args.to_file)

    if not from_path.exists():
        print(f"Error: source file not found: {args.from_file}", file=sys.stderr)
        sys.exit(1)
    if not to_path.exists():
        print(f"Error: target file not found: {args.to_file}", file=sys.stderr)
        sys.exit(1)

    # Both endpoints are rewritten by this command, so both are locked for
    # the whole plan: non-blocking, source then destination — a fixed order
    # that cannot deadlock against another move-to-file between the same
    # pair (or any other fastedit command holding one endpoint).
    with contextlib.ExitStack() as locks:
        locks.enter_context(_locked_for_edit(from_path))
        locks.enter_context(_locked_for_edit(to_path))
        _cmd_move_to_file_locked(args)


def _cmd_move_to_file_locked(args):
    """cmd_move_to_file's body, under both endpoints' cross-process locks."""
    from .inference.caller_safety import _find_project_root
    from .inference.move_to_file import move_to_file

    from_path = Path(args.from_file)
    to_path = Path(args.to_file)

    if not from_path.exists():
        print(f"Error: source file not found: {args.from_file}", file=sys.stderr)
        sys.exit(1)
    if not to_path.exists():
        print(f"Error: target file not found: {args.to_file}", file=sys.stderr)
        sys.exit(1)

    project_root = _find_project_root(from_path)

    try:
        plan = move_to_file(
            symbol=args.symbol,
            from_file=str(from_path),
            to_file=str(to_path),
            after=args.after,
            project_root=project_root,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(plan.message)



def _report_symbols_after_write(file_str: str, content: str, total_lines: int) -> None:
    """Print a symbol summary for a just-written file, or say why there isn't one.

    Shared by cmd_create and cmd_duplicate so both report the same way: a
    small file gets the inline preview, a large parseable file gets a
    tldr-structure summary, and an unparseable language just gets a note.
    """
    import json as json_mod
    import subprocess

    from .data_gen.ast_analyzer import detect_language

    language = detect_language(Path(file_str))
    if language is None:
        print(f"Note: {file_str} has no fastedit/tldr language support -- skipped the symbol check.")
        return

    if total_lines <= 100:
        print(_format_small_file(file_str, content, total_lines))
        return

    try:
        result = subprocess.run(
            ["tldr", "structure", file_str, "--format", "compact"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            print(f"Warning: tldr structure failed for {file_str}: {result.stderr.strip()}", file=sys.stderr)
            return
        data = json_mod.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Warning: tldr structure timed out for {file_str}", file=sys.stderr)
        return
    except FileNotFoundError:
        return
    except json_mod.JSONDecodeError as e:
        print(f"Warning: {e}", file=sys.stderr)
        return

    print(_format_structure(file_str, data, total_lines))

def cmd_create(args):
    """Create a new text file with the given content.

    Content comes from --content, --content-file (a path, or '-' for
    stdin), or stdin directly when neither flag is given. Refuses to
    overwrite an existing file unless --force is set, refuses a missing
    parent directory unless --parents is set, and refuses content the
    fastedit.filetype text/binary detector sniffs as binary
    (never by extension).
    """
    from .filetype import is_text_file
    from .mcp.backup import BackupStore, _atomic_write

    path = Path(args.file)
    if path.exists() and not args.force:
        print(f"Error: file already exists: {args.file} (use --force to overwrite)", file=sys.stderr)
        sys.exit(2)
    if not path.parent.exists():
        if not args.parents:
            print(f"Error: parent directory not found: {path.parent} (use --parents to create it)", file=sys.stderr)
            sys.exit(1)
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.content is not None and args.content_file is not None:
        print("Error: --content and --content-file are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if args.content is not None:
        raw = args.content.encode("utf-8")
    elif args.content_file is not None:
        raw = sys.stdin.buffer.read() if args.content_file == "-" else Path(args.content_file).read_bytes()
    else:
        raw = sys.stdin.buffer.read()

    detection = is_text_file(raw)
    if not detection.is_text:
        print(f"Error: refusing to create a binary file: {args.file} ({detection.reason})", file=sys.stderr)
        sys.exit(1)
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"Error: content for {args.file} is not valid UTF-8 text ({e})", file=sys.stderr)
        sys.exit(1)

    backups = BackupStore()
    _atomic_write(path, content, backups=backups)
    total_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    print(f"Created {args.file} ({total_lines} lines).")

    _report_symbols_after_write(args.file, content, total_lines)



def cmd_duplicate(args):
    """Duplicate a file byte-for-byte to a new path.

    Refuses when the source doesn't exist, when the destination already
    exists unless --force is set, or when the destination's parent
    directory is missing unless --parents is set. The content is copied
    as raw bytes with no decode step -- a byte-for-byte copy has nothing
    for a text/binary detector to protect against, so a binary source
    (images, archives, etc.) duplicates cleanly.
    """
    from .mcp.backup import BackupStore, _atomic_write

    src = Path(args.src)
    if not src.exists():
        print(f"Error: source file not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    raw = src.read_bytes()

    dst = Path(args.dst)
    if dst.exists() and not args.force:
        print(f"Error: file already exists: {args.dst} (use --force to overwrite)", file=sys.stderr)
        sys.exit(2)
    if not dst.parent.exists():
        if not args.parents:
            print(f"Error: parent directory not found: {dst.parent} (use --parents to create it)", file=sys.stderr)
            sys.exit(1)
        dst.parent.mkdir(parents=True, exist_ok=True)

    backups = BackupStore()
    _atomic_write(dst, raw, backups=backups)
    total_lines = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
    print(f"Duplicated {args.src} to {args.dst} ({total_lines} lines).")

    _report_symbols_after_write(args.dst, raw.decode("utf-8", errors="replace"), total_lines)

def cmd_split(args):
    """Split a file into per-chunk parts under --out, format-aware where possible."""
    import json as json_mod

    from .filetype import is_text_file
    from .split_join import (
        MANIFEST_NAME,
        SplitJoinError,
        detect_format,
        split_by_lines,
        split_csv_rows,
        split_json_array,
        split_markdown_headings,
        split_markup_top_level_children,
    )

    src = Path(args.file)
    if not src.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    raw = src.read_bytes()
    detection = is_text_file(raw)
    if not detection.is_text:
        print(f"Error: refusing to split a binary file: {args.file} ({detection.reason})", file=sys.stderr)
        sys.exit(1)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"Error: {args.file} is not valid UTF-8 text ({e})", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(src)
    ext = src.suffix
    manifest: dict = {"source_name": src.name, "format": fmt}

    def _write_parts(chunks: list[str]) -> list[str]:
        names = []
        for i, chunk in enumerate(chunks):
            name = f"part-{i:04d}{ext}"
            (out_dir / name).write_text(chunk, encoding="utf-8")
            names.append(name)
        return names

    try:
        if args.by == "element":
            if fmt == "json":
                prefix, elements, separators, suffix = split_json_array(text)
                manifest.update({
                    "mode": "element",
                    "joinable": True,
                    "parts": _write_parts(elements),
                    "json_prefix": prefix,
                    "json_separators": separators,
                    "json_suffix": suffix,
                })
            elif fmt in ("xml", "html"):
                children = split_markup_top_level_children(text)
                if not children:
                    print(f"Error: no top-level child elements found in {args.file}", file=sys.stderr)
                    sys.exit(1)
                manifest.update({"mode": "element", "joinable": False, "parts": _write_parts(children)})
                print(
                    "Warning: XML/HTML element splits are lossy (fragments lose ancestor "
                    "namespaces/xml:base/context) and read-only -- there is no `join` for this split.",
                    file=sys.stderr,
                )
            else:
                print(f"Error: --by element does not apply to a {fmt} file (only json/xml/html)", file=sys.stderr)
                sys.exit(1)
        elif args.by == "heading":
            if fmt != "markdown":
                print(f"Error: --by heading does not apply to a {fmt} file (only markdown/mdx)", file=sys.stderr)
                sys.exit(1)
            chunks = split_markdown_headings(text, args.level)
            manifest.update({"mode": "heading", "joinable": True, "level": args.level, "parts": _write_parts(chunks)})
        elif args.lines is not None:
            manifest.update({
                "mode": "lines", "joinable": True, "parts": _write_parts(split_by_lines(text, args.lines)),
            })
        elif args.rows is not None:
            if fmt not in ("csv", "tsv"):
                print(f"Error: --rows does not apply to a {fmt} file (only csv/tsv)", file=sys.stderr)
                sys.exit(1)
            manifest.update({
                "mode": "rows", "joinable": True, "parts": _write_parts(split_csv_rows(text, args.rows)),
            })
    except SplitJoinError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    (out_dir / MANIFEST_NAME).write_text(json_mod.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    n_parts = len(manifest["parts"])
    print(f"Split {args.file} into {n_parts} part(s) in {args.out}.")


def cmd_join(args):
    """Join split parts back into one file, inverting `split`'s mode."""
    import json as json_mod

    from .split_join import (
        MANIFEST_NAME,
        detect_format,
        join_csv_chunks,
        join_json_array,
    )

    inputs = [Path(p) for p in args.inputs]
    manifest = None
    if len(inputs) == 1 and inputs[0].is_dir():
        directory = inputs[0]
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.exists():
            print(f"Error: no {MANIFEST_NAME} found in {directory}", file=sys.stderr)
            sys.exit(1)
        manifest = json_mod.loads(manifest_path.read_text(encoding="utf-8"))
        part_paths = [directory / name for name in manifest["parts"]]
    else:
        part_paths = inputs
        if part_paths:
            manifest_path = part_paths[0].parent / MANIFEST_NAME
            if manifest_path.exists():
                manifest = json_mod.loads(manifest_path.read_text(encoding="utf-8"))

    for p in part_paths:
        if not p.exists():
            print(f"Error: part not found: {p}", file=sys.stderr)
            sys.exit(1)

    if manifest is not None and manifest.get("joinable") is False:
        mode = manifest.get("mode")
        fmt = manifest.get("format")
        print(
            f"Error: this split is not joinable (mode={mode!r}, format={fmt!r}) -- "
            "XML/HTML element splits are read-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read parts as raw bytes and write back as raw bytes: text-mode I/O here
    # would run universal-newline translation on read (CRLF/CR -> LF), silently
    # corrupting any part whose line endings are not bare LF. Bytes bypass that
    # translation entirely, so join is byte-exact regardless of line ending.
    chunks = [p.read_bytes() for p in part_paths]
    mode = manifest.get("mode") if manifest else None
    fmt = manifest.get("format") if manifest else detect_format(Path(args.out))

    if mode == "element" and fmt == "json":
        result = join_json_array(
            manifest["json_prefix"], chunks, manifest["json_separators"], manifest["json_suffix"],
        )
    elif fmt in ("csv", "tsv"):
        result = join_csv_chunks(chunks)
    else:
        result = b"".join(chunks)

    Path(args.out).write_bytes(result)
    print(f"Joined {len(part_paths)} part(s) into {args.out}.")


def _format_search_results(stdout: str, mode: str) -> str:
    text = stdout.strip()
    if text:
        return text
    if mode == "references":
        return "No references found."
    return "No results found."


def cmd_search(args):
    """Search codebase for functions, symbols, and references."""
    import subprocess

    if args.mode == "references":
        cmd = ["tldr", "references", args.query, args.path,
               "--format", "text", "--limit", str(args.top_k)]
        error_label = "references"
    else:
        cmd = ["tldr", "search", args.query, args.path,
               "--format", "text", "--top-k", str(args.top_k)]
        if args.mode == "regex":
            cmd.append("--regex")
        elif args.mode == "hybrid" and args.regex_filter:
            cmd.extend(["--hybrid", args.regex_filter])
        error_label = "search"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode != 0:
            print(f"Error: tldr {error_label} failed: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        print(_format_search_results(result.stdout, args.mode))
    except subprocess.TimeoutExpired:
        print(f"Error: {error_label} search timed out", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: tldr not found on PATH", file=sys.stderr)
        sys.exit(1)


def _decode_for_display(data: bytes, encoding: str | None) -> str:
    """Decode raw bytes FOR DISPLAY ONLY (undo/diff output) -- the result is
    never written anywhere. Prefers the codec ``read_source`` detected for
    the current file so a latin-1 backup shows its real characters; falls
    back to UTF-8 with errors="replace" when no codec is known (the current
    file no longer decodes), where replacement characters in a display diff
    are preferable to a crash. On disk, bytes are restored byte-for-byte.
    """
    return data.decode(encoding or "utf-8", errors="replace")


def cmd_diff(args):
    """Show unified diff between the last backup and the current file content."""
    import difflib

    from .io_utils import UnsupportedEncodingError, read_source
    from .mcp.backup import BackupStore

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    backups = BackupStore()

    if args.file not in backups:
        print(f"No backup recorded for {args.file}. Run an edit command first.")
        return

    # B38 layout: peek the NEWEST backup without popping -- diff must not
    # consume the undo history.
    backup_bytes = backups.peek(args.file)

    # Display-only decode (see _decode_for_display): both sides are decoded
    # with the codec the current file reads as, so a latin-1 file diffs
    # against its real characters; the bytes themselves are never rewritten.
    try:
        current, display_encoding = read_source(path)
    except UnsupportedEncodingError:
        current = path.read_text(encoding="utf-8", errors="replace")
        display_encoding = None
    backup_content = _decode_for_display(backup_bytes, display_encoding)

    if backup_content == current:
        print(f"No changes detected in {args.file}.")
        return

    diff = difflib.unified_diff(
        backup_content.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    print("".join(diff), end="")


def cmd_undo(args):
    """Revert the last edit to a file using BackupStore."""
    # The undo WRITES the target file, so it takes the same cross-process
    # edit lock as the verbs that created the backup.
    with _locked_for_edit(Path(args.file)):
        _cmd_undo_locked(args)


def _cmd_undo_locked(args):
    """cmd_undo's body, under the target file's cross-process edit lock."""
    import difflib

    from .io_utils import UnsupportedEncodingError, read_source
    from .mcp.backup import BackupStore, _atomic_write

    backups = BackupStore()
    path = Path(args.file)

    if args.file not in backups:
        print(f"Error: no undo history for {args.file}. Nothing to revert.", file=sys.stderr)
        sys.exit(1)

    # The current file is read for the DISPLAY diff only -- the restore
    # below writes raw bytes and no longer depends on any codec. A file
    # that no longer decodes (edited externally into something undecodable)
    # degrades the diff to replacement characters instead of crashing.
    if path.exists():
        try:
            current, display_encoding = read_source(path)
        except UnsupportedEncodingError:
            current = path.read_text(encoding="utf-8", errors="replace")
            display_encoding = None
    else:
        current, display_encoding = "", "utf-8"

    # B22/B38: backups are raw bytes; pop returns (and removes) the NEWEST
    # one, so repeated undos walk back one step at a time.
    backup_bytes = backups.pop(args.file)

    # Byte-for-byte restore WITHOUT passing backups -- no backup-of-backup
    # (no undo-of-undo). bytes content bypasses _atomic_write's str/BOM path
    # and is written exactly as stored.
    _atomic_write(path, backup_bytes)

    # Display-only decode of the popped bytes (see _decode_for_display):
    # the codec the current file reads as; never written back to disk.
    backup_text = _decode_for_display(backup_bytes, display_encoding)

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        backup_text.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    print(f"Reverted {args.file} to previous state.")
    print("".join(diff), end="")


def cmd_pull(args):
    """Pull the merge model from HuggingFace."""
    from .model_download import get_model_path
    path = get_model_path(model_name=args.model)
    print(f"Model ready at: {path}")


# ---------------------------------------------------------------------------
# Agent-skill installer (init)
# ---------------------------------------------------------------------------

_SKILL_NAME = "fastedit"
_SKILL_TIMEOUT_S = 300  # generous: npx's first run may download the skills CLI
_SKILL_TAIL_LINES = 15
# Presentation names for the success line; the raw agent id is used when the
# agent has no entry here (e.g. --skill-agent codex).
_AGENT_DISPLAY_NAMES = {"claude-code": "Claude Code"}


def _packaged_skill_bytes() -> bytes:
    """The agent skill content shipped inside the installed fastedit package.

    The wheel carries the repo's skills/fastedit/SKILL.md as package data at
    fastedit/skill/SKILL.md (byte-identity is drift-guarded by
    tests/test_fastedit_skill.py), so init never has to resolve a GitHub
    branch to know which skill content matches this install.
    """
    return (importlib.resources.files("fastedit") / "skill" / "SKILL.md").read_bytes()


def _stage_skill_tree(staging_root: Path) -> Path:
    """Write the packaged SKILL.md to <staging_root>/skills/fastedit/SKILL.md.

    Returns the staged skills/fastedit directory — the path handed to
    `npx skills add`. The directory form (a folder holding SKILL.md, named
    after the skill's frontmatter name) is the locally-verified install
    shape, and the skills/ level mirrors the repo layout the skills CLI
    discovers.
    """
    skill_dir = staging_root / "skills" / _SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_bytes(_packaged_skill_bytes())
    return skill_dir


def _skills_add_argv(skill_dir: Path, agent: str) -> list[str]:
    """The exact skills-CLI invocation `fastedit init` runs.

    The source is a STAGED LOCAL directory, never a GitHub shorthand: the
    shorthand resolves the fork's default branch (main), which still carries
    the legacy claude-skill content, and the tree-URL form fails outright in
    non-TTY mode. No --skill filter — the staged directory IS the skill.
    """
    return [
        "npx", "--yes", "skills", "add", str(skill_dir),
        "-g", "-a", agent, "-y",
    ]


def _skills_output_tail(*streams: str) -> str:
    """Last meaningful lines of the skills-CLI output, for the init report."""
    lines: list[str] = []
    for stream in streams:
        lines.extend((stream or "").splitlines())
    meaningful = [line for line in lines if line.strip()]
    return "\n".join(meaningful[-_SKILL_TAIL_LINES:])


def cmd_init(args):
    """One-shot environment setup: install the fastedit agent skill.

    The skill content SHIPS with the installed fastedit package: the wheel
    carries skills/fastedit/SKILL.md as package data at
    fastedit/skill/SKILL.md, byte-identical to the repo skill (drift-guarded
    by tests/test_fastedit_skill.py). cmd_init stages that packaged file into
    a FRESH temp directory laid out as <tmp>/skills/fastedit/SKILL.md and
    points the Vercel skills CLI (npx) at the staged directory, installing
    globally for the target coding agent (--skill-agent, default claude-code),
    then prints next-step guidance.

    Nothing is fetched from GitHub. The repo shorthand and tree-URL forms of
    `npx skills add` resolve the fork's DEFAULT branch (main), which still
    carries the legacy claude-skill content, so what an agent reads would not
    match the installed fastedit — and the tree-URL form fails outright in
    non-TTY mode. Staging the packaged copy removes the branch question
    entirely: the skill always matches the fastedit that ships it. The
    staging directory is removed on success and LEFT IN PLACE on failure, so
    the printed manual command is directly runnable.

    Failures are loud: npx missing, a staging failure, and a failing or
    timing-out skills-CLI run all exit 1 with guidance — an init that did
    nothing must say so.
    """
    import shutil
    import subprocess
    import tempfile

    agent = args.skill_agent
    display = _AGENT_DISPLAY_NAMES.get(agent, agent)

    if shutil.which("npx") is None:
        print(
            "Error: npx (Node.js) not found on PATH — the agent skill was NOT installed.",
            file=sys.stderr,
        )
        print(
            "Install Node.js (https://nodejs.org), then re-run: fastedit init",
            file=sys.stderr,
        )
        sys.exit(1)

    staging_root = Path(tempfile.mkdtemp(prefix="fastedit-init-skill-"))
    try:
        skill_dir = _stage_skill_tree(staging_root)
    except OSError as e:
        shutil.rmtree(staging_root, ignore_errors=True)
        print(
            f"Error: could not stage the packaged agent skill: {e} — "
            "the agent skill was NOT installed.",
            file=sys.stderr,
        )
        print(
            "The skill ships inside the fastedit package (fastedit/skill/SKILL.md); "
            "reinstall fastedits, then re-run: fastedit init",
            file=sys.stderr,
        )
        sys.exit(1)

    argv = _skills_add_argv(skill_dir, agent)
    manual = " ".join(argv)

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=_SKILL_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"Error: the skills CLI timed out after {_SKILL_TIMEOUT_S}s — "
            "the agent skill was NOT installed.",
            file=sys.stderr,
        )
        print(f"Retry, or run the installer manually: {manual}", file=sys.stderr)
        print(f"(the staged skill directory was left in place: {skill_dir})", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: could not run npx: {e}", file=sys.stderr)
        print(f"Run the installer manually: {manual}", file=sys.stderr)
        print(f"(the staged skill directory was left in place: {skill_dir})", file=sys.stderr)
        sys.exit(1)

    tail = _skills_output_tail(result.stdout, result.stderr)
    if result.returncode != 0:
        # Fail loud with the tool's own output so the user sees what happened.
        if tail:
            print(tail, file=sys.stderr)
        print(
            f"Error: the skills CLI exited with code {result.returncode} — "
            "the agent skill was NOT installed.",
            file=sys.stderr,
        )
        print(
            f"Run the installer manually to see the live output: {manual}",
            file=sys.stderr,
        )
        print(f"(the staged skill directory was left in place: {skill_dir})", file=sys.stderr)
        sys.exit(1)

    shutil.rmtree(staging_root, ignore_errors=True)

    if tail:
        print(tail)
    print(
        f"Agent skill installed (global, {display}). "
        "Next: fastedit pull --model mlx-8bit (Apple Silicon) · fastedit doctor"
    )
    print("Optional MCP entry: fastedit mcp-install")


# ---------------------------------------------------------------------------
# Argparse setup and main dispatch
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="fastedit",
        description="FastEdit — AST-aware code editing via CLI",
        # RawDescriptionHelpFormatter: keeps the option list formatting while
        # printing the guide (cli_help.EPILOG) verbatim, un-wrapped.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    # metavar: the 20-command choices token is unbreakable for argparse and
    # renders as a single long line in usage; "command" keeps the
    # usage lines short (the epilog's COMMANDS section lists every command).
    sub = parser.add_subparsers(dest="command", metavar="command")

    # --- read (no model) ---
    read_p = sub.add_parser("read", help="Show file structure (functions, classes, line ranges)")
    read_p.add_argument("file", help="Path to source file")

    # --- search (no model) ---
    search_p = sub.add_parser("search", help="Search codebase for symbols and functions")
    search_p.add_argument("query", help="Search query or symbol name")
    search_p.add_argument("path", nargs="?", default=".", help="Directory to search (default: .)")
    search_p.add_argument(
        "--mode",
        choices=["search", "regex", "hybrid", "references"],
        default="search",
        help="Search mode (default: search)",
    )
    search_p.add_argument("--top-k", type=int, default=10, help="Max results (default: 10)")
    search_p.add_argument("--regex-filter", default="", help="Regex filter for hybrid mode")

    # --- diff (no model) ---
    diff_p = sub.add_parser("diff", help="Show diff between last backup and current file")
    diff_p.add_argument("file", help="Path to source file")

    # --- edit (model) ---
    edit_p = sub.add_parser(
        "edit",
        help="Apply an edit snippet to a file "
             "(shows a caller-impact note when --replace changes a signature)",
    )
    edit_p.add_argument("file", help="Path to source file")
    edit_p.add_argument("--snippet", required=True, help="Edit snippet or '-' for stdin")
    edit_p.add_argument("--after", default="", help="Insert new code after this symbol")
    edit_p.add_argument("--replace", default="", help="Replace this symbol with the snippet")
    edit_p.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    edit_p.add_argument("--model-path", default=None, help="MLX model path (overrides FASTEDIT_MODEL_PATH)")
    edit_p.add_argument("--api-base", default=None, help="vLLM API base URL")
    edit_p.add_argument("--api-model", default=None, help="vLLM model name")

    # --- batch-edit (model) ---
    be_p = sub.add_parser("batch-edit", help="Apply multiple edits to one file")
    be_p.add_argument("file", help="Path to source file")
    be_p.add_argument(
        "--edits", required=True,
        help=(
            'JSON list of edits. Each item: {"snippet": "...", "after": "sym"} '
            'or {"snippet": "...", "replace": "sym"}. Use \'-\' for stdin.'
        ),
    )
    be_p.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    be_p.add_argument("--model-path", default=None)
    be_p.add_argument("--api-base", default=None)
    be_p.add_argument("--api-model", default=None)

    # --- multi-edit (model) ---
    me_p = sub.add_parser("multi-edit", help="Apply edits across multiple files")
    me_p.add_argument(
        "--file-edits", required=True,
        help=(
            'JSON list. Each item: {"file_path": "...", "edits": [...]}. '
            'Use \'-\' for stdin.'
        ),
    )
    me_p.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    me_p.add_argument("--model-path", default=None)
    me_p.add_argument("--api-base", default=None)
    me_p.add_argument("--api-model", default=None)

    # --- delete (no model) ---
    del_p = sub.add_parser(
        "delete",
        help="Delete a function/class/method by name "
             "(refuses if cross-file callers exist; use --force to override)",
    )
    del_p.add_argument("file", help="Path to source file")
    del_p.add_argument("symbol", help="Symbol name to delete (e.g. 'my_func' or 'MyClass.method')")
    del_p.add_argument(
        "--force", action="store_true",
        help="Delete even if other files still reference the symbol "
             "(skips the cross-file caller-safety check).",
    )

    # --- move (no model) ---
    mv_p = sub.add_parser("move", help="Move a symbol to after another symbol")
    mv_p.add_argument("file", help="Path to source file")
    mv_p.add_argument("symbol", help="Symbol to move")
    mv_p.add_argument("--after", required=True, help="Move after this symbol")

    # --- rename (no model) ---
    rn_p = sub.add_parser(
        "rename",
        help="Rename all AST-verified occurrences of a symbol in a single "
             "file (skips strings/comments; supports --dry-run)",
    )
    rn_p.add_argument("file", help="Path to source file")
    rn_p.add_argument("old_name", help="Current symbol name")
    rn_p.add_argument("new_name", help="New symbol name")
    rn_p.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would change without writing",
    )

    # rename-all (cross-file rename, no model)
    ra_p = sub.add_parser(
        "rename-all",
        help="Rename a symbol across every supported file in a directory",
    )
    ra_p.add_argument("root", help="Directory to walk")
    ra_p.add_argument("old_name", help="Current symbol name")
    ra_p.add_argument("new_name", help="New symbol name")
    ra_p.add_argument(
        "--dry-run", action="store_true",
        help="Preview which files would change without writing",
    )
    ra_p.add_argument(
        "--only", choices=["class", "function", "method", "variable"], default=None,
        help="Restrict rename to targets whose definition kind matches "
             "(uses tldr for AST-verified lookup).",
    )

    # --- move-to-file (no model) ---
    mtf_p = sub.add_parser(
        "move-to-file",
        help="Move a symbol from one file to another and rewrite imports",
    )
    mtf_p.add_argument("symbol", help="Symbol to move")
    mtf_p.add_argument("from_file", help="Source file (must contain the symbol)")
    mtf_p.add_argument("to_file", help="Destination file (must exist, same language)")
    mtf_p.add_argument(
        "--after",
        default=None,
        help="Insert after this symbol in the destination (default: end of file)",
    )
    mtf_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the plan without writing any files",
    )

    # --- create (no model) ---
    create_p = sub.add_parser(
        "create",
        help="Create a new source file with the given content",
    )
    create_p.add_argument("file", help="Path of the file to create")
    create_p.add_argument(
        "--content",
        default=None,
        help="File content (alternative: --content-file, or stdin if neither is given)",
    )
    create_p.add_argument(
        "--content-file",
        default=None,
        help="Read file content from this path, or '-' for stdin",
    )
    create_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the file if it already exists",
    )
    create_p.add_argument(
        "--parents",
        action="store_true",
        help="Create missing parent directories",
    )

    # --- duplicate (no model) ---
    duplicate_p = sub.add_parser(
        "duplicate",
        help="Duplicate a file to a new path, byte-for-byte",
    )
    duplicate_p.add_argument("src", help="Path of the file to duplicate")
    duplicate_p.add_argument("dst", help="Destination path")
    duplicate_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists",
    )
    duplicate_p.add_argument(
        "--parents",
        action="store_true",
        help="Create missing destination parent directories",
    )

    # --- split (no model) ---
    split_p = sub.add_parser(
        "split",
        help="Split a file into per-chunk parts, format-aware where possible",
    )
    split_p.add_argument("file", help="Path to the file to split")
    split_p.add_argument("--out", required=True, help="Output directory for the parts")
    split_mode = split_p.add_mutually_exclusive_group(required=True)
    split_mode.add_argument(
        "--by", choices=["element", "heading"], default=None,
        help="Split by structural element (json array / xml / html) or by heading (markdown)",
    )
    split_mode.add_argument(
        "--lines", type=int, default=None,
        help="Split into chunks of N lines (any text file)",
    )
    split_mode.add_argument(
        "--rows", type=int, default=None,
        help="Split into chunks of N data rows, repeating the header (csv/tsv)",
    )
    split_p.add_argument(
        "--level", type=int, default=2,
        help="Heading level for --by heading, e.g. 2 for '## ' (default: 2)",
    )

    # --- join (no model) ---
    join_p = sub.add_parser(
        "join",
        help="Join split parts back into one file (inverts 'split')",
    )
    join_p.add_argument(
        "inputs", nargs="+",
        help="A split output directory, or explicit part files in order",
    )
    join_p.add_argument("-o", "--out", required=True, help="Destination file to write")

    # --- undo (no model) ---
    undo_p = sub.add_parser("undo", help="Revert the last edit to a file")
    undo_p.add_argument("file", help="Path to source file")

    # init (one-shot setup: installs the agent skill via the skills CLI)
    init_p = sub.add_parser(
        "init",
        help="Install the fastedit agent skill for your coding agent (npx skills CLI)",
    )
    init_p.add_argument(
        "--skill-agent",
        default="claude-code",
        help="Coding agent to install the skill for (default: claude-code)",
    )

    # pull
    pull_p = sub.add_parser("pull", help="Pull the merge model from HuggingFace (~3GB)")
    pull_p.add_argument("--model", required=True, choices=["mlx-8bit", "bf16"],
                        help="Model to download. Use mlx-8bit on Apple Silicon (MLX), bf16 on Linux GPU (vLLM).")

    # doctor (diagnostics)
    sub.add_parser("doctor", help="Run self-diagnostic and report install health")

    # mcp-install (write Claude Code MCP config)
    mcp_install_p = sub.add_parser(
        "mcp-install",
        help="Install the fastedit MCP entry in Claude Code config",
    )
    mcp_install_p.add_argument(
        "--scope", choices=["user", "project"], default="user",
        help="Where to write the entry (default: user = ~/.claude.json)",
    )

    args = parser.parse_args()

    if args.command == "read":
        cmd_read(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "edit":
        cmd_edit(args)
    elif args.command == "batch-edit":
        cmd_batch_edit(args)
    elif args.command == "multi-edit":
        cmd_multi_edit(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "move":
        cmd_move(args)
    elif args.command == "rename":
        cmd_rename(args)
    elif args.command == "rename-all":
        cmd_rename_all(args)
    elif args.command == "move-to-file":
        cmd_move_to_file(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "duplicate":
        cmd_duplicate(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "join":
        cmd_join(args)
    elif args.command == "undo":
        cmd_undo(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "pull":
        cmd_pull(args)
    elif args.command == "doctor":
        from .doctor import run_doctor
        sys.exit(run_doctor())
    elif args.command == "mcp-install":
        from .mcp_install import install_mcp_config
        sys.exit(install_mcp_config(args.scope))
    else:
        parser.print_help()

    # Passive update notice on exit. Silent when up-to-date, network-down,
    # or FASTEDIT_NO_UPDATE_CHECK=1. Runs after the command so it never
    # delays user-visible output.
    # Best-effort update notice -- an infra hiccup here must never fail the
    # user's already-completed command, hence the blanket suppression.
    with contextlib.suppress(Exception):
        from .update_check import get_update_notice
        notice = get_update_notice()
        if notice:
            sys.stderr.write("\n" + notice + "\n")


if __name__ == "__main__":
    main()
