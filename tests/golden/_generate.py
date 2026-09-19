"""Golden-case generator for the B3 per-format golden matrix (req. 3 + C1).

Committed generator: `uv run python tests/golden/_generate.py` re-writes
every fixture under ``tests/golden/<lang>/``.

THE ORACLE IS INDEPENDENT OF FASTEDIT. Expected outputs are computed by
explicit line-splice arithmetic against the line indices declared in the
LANGUAGES table below — this module never imports fastedit, never runs its
pipeline, and never consults a parser. The oracle encodes the DOCUMENTED
splice semantics of the deterministic paths as plain line arithmetic:

  insert_after(symbol)
      Splice the snippet lines (preservation-marker lines dropped — they
      are merge directives, never content) after the anchor symbol's last
      line, wrapped in one blank separator line on each side where the
      adjacent original line is non-blank.

  replace_symbol(span)
      Replace the span's lines with the effective snippet: the span's own
      leading signature lines when the op declares ``prepend_signature_lines``
      (the pipeline auto-prepends a signature the snippet omitted), then the
      snippet lines minus marker lines, each newline-terminated.

  delete_symbol(span)
      Remove the span's lines, then consume the blank lines that separated
      the deleted symbol from whatever follows (the leading blanks already
      preserved before the span become the new separator).

Every declared line index is cross-checked against the authored original
(``span_head`` / ``span_tail`` / ``anchor_head`` / ``anchor_tail``
expectations) so the table cannot silently drift from the fixtures, and
every snippet is asserted pre-aligned to its target's FIRST-line indent
(the pipeline's indent alignment is a no-op on every golden op by
construction).

The manifest-driven runner (tests/test_golden_matrix.py) re-derives the
expected bytes from the manifest with THESE SAME oracle functions and
compares them against the committed expected files, so a stale artifact
fails loudly instead of pinning wrong behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent

# A preservation-marker line (canonical long forms; see
# fastedit.inference.markers) is a merge directive: the oracle drops it
# from snippet content exactly as the pipeline does.
_MARKER_LINE = re.compile(r"^(?:#|//)\s*\.\.\.\s*existing code\s*\.\.\.\s*$")


def is_marker_line(line: str) -> bool:
    """True for a preservation-marker directive line (never content)."""
    return bool(_MARKER_LINE.match(line.strip()))


def _terminated(snippet: str) -> list[str]:
    """Snippet text -> newline-terminated lines, markers dropped."""
    body = snippet.rstrip("\n")
    return [
        line + "\n"
        for line in body.split("\n")
        if not is_marker_line(line)
    ]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def oracle_insert_after(
    lines: list[str], anchor_end_line: int, snippet: str,
) -> list[str]:
    """Splice the snippet after the anchor's last line (blank separators)."""
    before = lines[:anchor_end_line]
    after = lines[anchor_end_line:]
    kept = _terminated(snippet)
    separator = ["\n"] if before and before[-1].strip() != "" else []
    trailing = ["\n"] if after and after[0].strip() != "" else []
    return before + separator + kept + trailing + after


def oracle_replace_span(
    lines: list[str],
    start_line: int,
    end_line: int,
    snippet: str,
    prepend_signature_lines: int = 0,
) -> list[str]:
    """Replace the span's lines with the effective snippet lines."""
    replacement = list(lines[start_line - 1 : start_line - 1 + prepend_signature_lines])
    replacement += _terminated(snippet)
    return lines[: start_line - 1] + replacement + lines[end_line:]


def oracle_delete_span(
    lines: list[str], start_line: int, end_line: int,
) -> list[str]:
    """Remove the span plus the blank lines that separated it from the rest."""
    end_idx = end_line
    while end_idx < len(lines) and lines[end_idx].strip() == "":
        end_idx += 1
    return lines[: start_line - 1] + lines[end_idx:]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# The golden matrix. Fixture content and oracle indices are HAND-DECLARED:
# the generator verifies every index against the authored original before
# writing anything.
# ---------------------------------------------------------------------------

LANGUAGES: list[dict] = [
    # --- the 15 original AST languages -----------------------------------
    {
        "language": "python", "ext": "py", "filename": "original.py",
        "symbol_semantics": (
            "top-level functions, classes, methods (parent-qualified) — "
            "module-level UPPER_CASE assignments are deliberately not symbols "
            "(see the _CONST_LIKE_NODE_TYPES note in ast_utils)"
        ),
        "original": '''"""Sample module."""
CONSTANT = 42


def alpha(x):
    return x + 1


class Store:
    def save(self):
        return "saved"


def beta(y):
    return y * 2
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "def gamma(z):\n    return z - 1\n",
                "anchor_start_line": 5, "anchor_head": "def alpha(x):",
                "anchor_end_line": 6, "anchor_tail": "    return x + 1",
                "snippet_head": "def gamma(z):",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "def beta(y):\n    return y * 3\n",
                "start_line": 14, "end_line": 15,
                "span_head": "def beta(y):", "span_tail": "    return y * 2",
                "snippet_head": "def beta(y):",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 5, "end_line": 6,
                "span_head": "def alpha(x):", "span_tail": "    return x + 1",
            },
        ],
        "gigo": {
            "defect": "syntax error in an unrelated function (def broken(:) — EDIT-NOT-CORRECT: preserved byte-exact",
            "defect_lines": [4, 5],
            "original": '''"""Sample module."""


def broken(:
    return oops


def alpha(x):
    return x + 1


def beta(y):
    return y * 2
''',
            "op": {
                "op": "insert_after", "symbol": "beta",
                "snippet": "def gamma(z):\n    return z - 1\n",
                "anchor_start_line": 12, "anchor_head": "def beta(y):",
                "anchor_end_line": 13, "anchor_tail": "    return y * 2",
                "snippet_head": "def gamma(z):",
            },
        },
    },
    {
        "language": "javascript", "ext": "js", "filename": "original.js",
        "symbol_semantics": "functions, classes, methods, top-level const/let (constants)",
        "original": '''const LIMIT = 10;


function alpha(x) {
    return x + 1;
}


class Cart {
    add(item) {
        this.items.push(item);
    }
}


function beta(y) {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "function gamma(z) {\n    return z - 1;\n}\n",
                "anchor_start_line": 4, "anchor_head": "function alpha(x) {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": "function gamma(z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "function beta(y) {\n    return y * 3;\n}\n",
                "start_line": 16, "end_line": 18,
                "span_head": "function beta(y) {", "span_tail": "}",
                "snippet_head": "function beta(y) {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT = 10;", "span_tail": "const LIMIT = 10;",
            },
        ],
    },
    {
        "language": "typescript", "ext": "ts", "filename": "original.ts",
        "symbol_semantics": "functions, classes, interfaces, method signatures, top-level const/let",
        "original": '''const LIMIT: number = 10;


function alpha(x: number): number {
    return x + 1;
}


interface Repo {
    save(): boolean;
}


function beta(y: number): number {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "function gamma(z: number): number {\n    return z - 1;\n}\n",
                "anchor_start_line": 4,
                "anchor_head": "function alpha(x: number): number {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": "function gamma(z: number): number {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "function beta(y: number): number {\n    return y * 3;\n}\n",
                "start_line": 14, "end_line": 16,
                "span_head": "function beta(y: number): number {", "span_tail": "}",
                "snippet_head": "function beta(y: number): number {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT: number = 10;",
                "span_tail": "const LIMIT: number = 10;",
            },
        ],
    },
    {
        "language": "tsx", "ext": "tsx", "filename": "original.tsx",
        "symbol_semantics": "components (functions), top-level const/let",
        "original": '''const LIMIT = 10;


function Badge({ label }: { label: string }) {
    return <span className="badge">{label}</span>;
}


function Panel({ title }: { title: string }) {
    return (
        <div className="panel">
            <Badge label={title} />
        </div>
    );
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Badge",
                "snippet": 'function Tag({ label }: { label: string }) {\n    return <span className="tag">{label}</span>;\n}\n',
                "anchor_start_line": 4,
                "anchor_head": "function Badge({ label }: { label: string }) {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": 'function Tag({ label }: { label: string }) {',
            },
            {
                "op": "replace_symbol", "symbol": "Badge", "path": "direct_swap",
                "snippet": 'function Badge({ label }: { label: string }) {\n    return <span className="tag">{label}</span>;\n}\n',
                "start_line": 4, "end_line": 6,
                "span_head": "function Badge({ label }: { label: string }) {",
                "span_tail": "}",
                "snippet_head": "function Badge({ label }: { label: string }) {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT = 10;", "span_tail": "const LIMIT = 10;",
            },
        ],
    },
    {
        "language": "rust", "ext": "rs", "filename": "original.rs",
        "symbol_semantics": "functions, structs/enums/traits/impls, const/static items",
        "original": '''const LIMIT: i32 = 10;

fn alpha(x: i32) -> i32 {
    x + 1
}

struct Item {
    name: String,
}

fn beta(y: i32) -> i32 {
    y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "fn gamma(z: i32) -> i32 {\n    z - 1\n}\n",
                "anchor_start_line": 3, "anchor_head": "fn alpha(x: i32) -> i32 {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "fn gamma(z: i32) -> i32 {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "fn beta(y: i32) -> i32 {\n    y * 3\n}\n",
                "start_line": 11, "end_line": 13,
                "span_head": "fn beta(y: i32) -> i32 {", "span_tail": "}",
                "snippet_head": "fn beta(y: i32) -> i32 {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT: i32 = 10;",
                "span_tail": "const LIMIT: i32 = 10;",
            },
        ],
    },
    {
        "language": "go", "ext": "go", "filename": "original.go",
        "symbol_semantics": "functions and methods, type declarations, var/const declarations",
        "original": '''package main

const Limit = 10

func Alpha(x int) int {
\treturn x + 1
}

type Item struct {
\tName string
}

func Beta(y int) int {
\treturn y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Alpha",
                "snippet": "func Gamma(z int) int {\n\treturn z - 1\n}\n",
                "anchor_start_line": 5, "anchor_head": "func Alpha(x int) int {",
                "anchor_end_line": 7, "anchor_tail": "}",
                "snippet_head": "func Gamma(z int) int {",
            },
            {
                "op": "replace_symbol", "symbol": "Beta", "path": "direct_swap",
                "snippet": "func Beta(y int) int {\n\treturn y * 3\n}\n",
                "start_line": 13, "end_line": 15,
                "span_head": "func Beta(y int) int {", "span_tail": "}",
                "snippet_head": "func Beta(y int) int {",
            },
            {
                "op": "delete_symbol", "symbol": "Limit",
                "start_line": 3, "end_line": 3,
                "span_head": "const Limit = 10", "span_tail": "const Limit = 10",
            },
        ],
    },
    {
        "language": "java", "ext": "java", "filename": "original.java",
        "symbol_semantics": (
            "classes/interfaces/enums, methods and constructors, fields "
            "(constants). The replace op passes a body-only snippet and the "
            "pipeline prepends the signature line from the AST"
        ),
        "original": '''public class Sample {
    private static final int LIMIT = 10;

    public static int alpha(int x) {
        return x + 1;
    }

    public static int beta(int y) {
        return y * 2;
    }
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "    public static int gamma(int z) {\n        return z - 1;\n    }\n",
                "anchor_start_line": 4,
                "anchor_head": "    public static int alpha(int x) {",
                "anchor_end_line": 6, "anchor_tail": "    }",
                "snippet_head": "    public static int gamma(int z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "        return y * 3;\n    }\n",
                "start_line": 8, "end_line": 10,
                "span_head": "    public static int beta(int y) {",
                "span_tail": "    }",
                "snippet_head": "        return y * 3;",
                "prepend_signature_lines": 1,
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 2, "end_line": 2,
                "span_head": "    private static final int LIMIT = 10;",
                "span_tail": "    private static final int LIMIT = 10;",
            },
        ],
    },
    {
        "language": "c", "ext": "c", "filename": "original.c",
        "symbol_semantics": "functions, struct/enum specifiers; preproc #define is not a symbol",
        "original": '''#include <stdio.h>

int alpha(int x) {
    return x + 1;
}

struct Item {
    int size;
};

int beta(int y) {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "int gamma(int z) {\n    return z - 1;\n}\n",
                "anchor_start_line": 3, "anchor_head": "int alpha(int x) {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "int gamma(int z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "int beta(int y) {\n    return y * 3;\n}\n",
                "start_line": 11, "end_line": 13,
                "span_head": "int beta(int y) {", "span_tail": "}",
                "snippet_head": "int beta(int y) {",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 3, "end_line": 5,
                "span_head": "int alpha(int x) {", "span_tail": "}",
            },
        ],
    },
    {
        "language": "cpp", "ext": "cpp", "filename": "original.cpp",
        "symbol_semantics": "functions, class/struct specifiers",
        "original": '''#include <string>

class Cart {
public:
    void add(int item);
};

int alpha(int x) {
    return x + 1;
}

int beta(int y) {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "int gamma(int z) {\n    return z - 1;\n}\n",
                "anchor_start_line": 8, "anchor_head": "int alpha(int x) {",
                "anchor_end_line": 10, "anchor_tail": "}",
                "snippet_head": "int gamma(int z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "int beta(int y) {\n    return y * 3;\n}\n",
                "start_line": 12, "end_line": 14,
                "span_head": "int beta(int y) {", "span_tail": "}",
                "snippet_head": "int beta(int y) {",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 8, "end_line": 10,
                "span_head": "int alpha(int x) {", "span_tail": "}",
            },
        ],
    },
    {
        "language": "ruby", "ext": "rb", "filename": "original.rb",
        "symbol_semantics": "methods and singleton methods, classes and modules",
        "original": '''LIMIT = 10

def alpha(x)
  x + 1
end

class Cart
  def add(item)
    item
  end
end

def beta(y)
  y * 2
end
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "def gamma(z)\n  z - 1\nend\n",
                "anchor_start_line": 3, "anchor_head": "def alpha(x)",
                "anchor_end_line": 5, "anchor_tail": "end",
                "snippet_head": "def gamma(z)",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "def beta(y)\n  y * 3\nend\n",
                "start_line": 13, "end_line": 15,
                "span_head": "def beta(y)", "span_tail": "end",
                "snippet_head": "def beta(y)",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 3, "end_line": 5,
                "span_head": "def alpha(x)", "span_tail": "end",
            },
        ],
    },
    {
        "language": "swift", "ext": "swift", "filename": "original.swift",
        "symbol_semantics": "functions and initializers, classes/structs/protocols, property declarations",
        "original": '''let limit = 10

func alpha(x: Int) -> Int {
    return x + 1
}

struct Item {
    var name: String
}

func beta(y: Int) -> Int {
    return y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "func gamma(z: Int) -> Int {\n    return z - 1\n}\n",
                "anchor_start_line": 3, "anchor_head": "func alpha(x: Int) -> Int {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "func gamma(z: Int) -> Int {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "func beta(y: Int) -> Int {\n    return y * 3\n}\n",
                "start_line": 11, "end_line": 13,
                "span_head": "func beta(y: Int) -> Int {", "span_tail": "}",
                "snippet_head": "func beta(y: Int) -> Int {",
            },
            {
                "op": "delete_symbol", "symbol": "limit",
                "start_line": 1, "end_line": 1,
                "span_head": "let limit = 10", "span_tail": "let limit = 10",
            },
        ],
    },
    {
        "language": "kotlin", "ext": "kt", "filename": "original.kt",
        "symbol_semantics": "functions, classes/objects, property declarations (constants)",
        "original": '''const val LIMIT = 10

fun alpha(x: Int): Int {
    return x + 1
}

class Cart {
    fun add(item: Int): Int {
        return item
    }
}

fun beta(y: Int): Int {
    return y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "fun gamma(z: Int): Int {\n    return z - 1\n}\n",
                "anchor_start_line": 3, "anchor_head": "fun alpha(x: Int): Int {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "fun gamma(z: Int): Int {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "fun beta(y: Int): Int {\n    return y * 3\n}\n",
                "start_line": 13, "end_line": 15,
                "span_head": "fun beta(y: Int): Int {", "span_tail": "}",
                "snippet_head": "fun beta(y: Int): Int {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const val LIMIT = 10",
                "span_tail": "const val LIMIT = 10",
            },
        ],
    },
    {
        "language": "c_sharp", "ext": "cs", "filename": "original.cs",
        "symbol_semantics": (
            "classes/interfaces/structs, methods and constructors, fields "
            "(constants). Allman braces: the replace op passes a body-only "
            "snippet and the pipeline prepends the 2-line signature from the AST"
        ),
        "original": '''public class Sample
{
    private const int Limit = 10;

    public static int Alpha(int x)
    {
        return x + 1;
    }

    public static int Beta(int y)
    {
        return y * 2;
    }
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Alpha",
                "snippet": "    public static int Gamma(int z)\n    {\n        return z - 1;\n    }\n",
                "anchor_start_line": 5,
                "anchor_head": "    public static int Alpha(int x)",
                "anchor_end_line": 8, "anchor_tail": "    }",
                "snippet_head": "    public static int Gamma(int z)",
            },
            {
                "op": "replace_symbol", "symbol": "Beta", "path": "direct_swap",
                "snippet": "        return y * 3;\n    }\n",
                "start_line": 10, "end_line": 13,
                "span_head": "    public static int Beta(int y)",
                "span_tail": "    }",
                "snippet_head": "        return y * 3;",
                "prepend_signature_lines": 2,
            },
            {
                "op": "delete_symbol", "symbol": "Limit",
                "start_line": 3, "end_line": 3,
                "span_head": "    private const int Limit = 10;",
                "span_tail": "    private const int Limit = 10;",
            },
        ],
    },
    {
        "language": "php", "ext": "php", "filename": "original.php",
        "symbol_semantics": (
            "functions, classes/interfaces/traits, methods, const declarations. "
            "The grammar parses script content only after <?php, so a "
            "standalone-parsed redefinition snippet cannot serve direct-swap; "
            "the replace op uses the marker-bearing text-match form with an "
            "assignment body (unique replacement key)"
        ),
        "original": '''<?php

const LIMIT = 10;

function alpha($x) {
    return $x + 1;
}

function beta($y) {
    $result = $y * 2;
    return $result;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "function gamma($z) {\n    return $z - 1;\n}\n",
                "anchor_start_line": 5, "anchor_head": "function alpha($x) {",
                "anchor_end_line": 7, "anchor_tail": "}",
                "snippet_head": "function gamma($z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "text_match",
                "snippet": (
                    "function beta($y) {\n"
                    "// ... existing code ...\n"
                    "    $result = $y * 3;\n"
                    "    return $result;\n"
                    "// ... existing code ...\n"
                    "}\n"
                ),
                "start_line": 9, "end_line": 12,
                "span_head": "function beta($y) {",
                "span_tail": "}",
                "snippet_head": "function beta($y) {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 3, "end_line": 3,
                "span_head": "const LIMIT = 10;", "span_tail": "const LIMIT = 10;",
            },
        ],
    },
    {
        "language": "elixir", "ext": "ex", "filename": "original.ex",
        "symbol_semantics": "defmodule (module), def/defp functions (methods inside a module)",
        "original": '''defmodule Sample do
  @limit 10

  def alpha(x) do
    x + 1
  end

  def beta(y) do
    y * 2
  end
end
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "  def gamma(z) do\n    z - 1\n  end\n",
                "anchor_start_line": 4, "anchor_head": "  def alpha(x) do",
                "anchor_end_line": 6, "anchor_tail": "  end",
                "snippet_head": "  def gamma(z) do",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "  def beta(y) do\n    y * 3\n  end\n",
                "start_line": 8, "end_line": 10,
                "span_head": "  def beta(y) do", "span_tail": "  end",
                "snippet_head": "  def beta(y) do",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 4, "end_line": 6,
                "span_head": "  def alpha(x) do", "span_tail": "  end",
            },
        ],
    },
    # --- the 10 B2/B3 hard-dependency data/config/markup formats ---------
    {
        "language": "html", "ext": "html", "filename": "original.html",
        "symbol_semantics": (
            "elements carrying an `id` attribute, named by the attribute VALUE "
            "(`id` is HTML's naming mechanism; tag names repeat and address "
            "nothing). Symbols nest. A replace= snippet that restates only "
            "part of the element would be an insert by the text-match editor's "
            "preserve-by-default semantics, so the golden replace op is a "
            "complete element re-definition served by the direct-swap span swap"
        ),
        "original": '''<!DOCTYPE html>
<html lang="en">
<head>
  <title>Sample</title>
</head>
<body>
  <section id="intro">
    <h1>Intro</h1>
    <p>Welcome paragraph.</p>
  </section>
  <section id="details">
    <h1>Details</h1>
    <p>Detail paragraph.</p>
  </section>
</body>
</html>
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "intro",
                "snippet": '  <section id="extra">\n    <p>Extra paragraph.</p>\n  </section>\n',
                "anchor_start_line": 7, "anchor_head": '  <section id="intro">',
                "anchor_end_line": 10, "anchor_tail": "  </section>",
                "snippet_head": '  <section id="extra">',
            },
            {
                "op": "replace_symbol", "symbol": "details", "path": "direct_swap",
                "snippet": '  <section id="details" class="wide">\n    <h2>Details</h2>\n    <p>Expanded detail paragraph.</p>\n  </section>\n',
                "start_line": 11, "end_line": 14,
                "span_head": '  <section id="details">',
                "span_tail": "  </section>",
                "snippet_head": '  <section id="details" class="wide">',
            },
            {
                "op": "delete_symbol", "symbol": "intro",
                "start_line": 7, "end_line": 10,
                "span_head": '  <section id="intro">', "span_tail": "  </section>",
            },
        ],
        "gigo": {
            "defect": "valueless attribute (attr=) in an unrelated div — the one html breakage the grammar trips on — preserved byte-exact",
            "defect_lines": [4, 4],
            "original": '''<!DOCTYPE html>
<html lang="en">
<body>
  <div attr=></div>
  <section id="intro">
    <h1>Intro</h1>
  </section>
  <section id="details">
    <h1>Details</h1>
  </section>
</body>
</html>
''',
            "op": {
                "op": "insert_after", "symbol": "details",
                "snippet": '  <section id="extra">\n    <p>Extra paragraph.</p>\n  </section>\n',
                "anchor_start_line": 8, "anchor_head": '  <section id="details">',
                "anchor_end_line": 10, "anchor_tail": "  </section>",
                "snippet_head": '  <section id="extra">',
            },
        },
    },
    {
        "language": "xml", "ext": "xml", "filename": "original.xml",
        "symbol_semantics": (
            "every element, named by its tag name; sibling repeats are refused "
            "as ambiguous (fail loud, never first-match). Symbols nest"
        ),
        "original": '''<?xml version="1.0"?>
<config>
  <server>host1</server>
  <cache enabled="true">redis</cache>
  <limits>
    <timeout>30</timeout>
  </limits>
</config>
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "server",
                "snippet": "  <replica>host2</replica>\n",
                "anchor_start_line": 3, "anchor_head": "  <server>host1</server>",
                "anchor_end_line": 3, "anchor_tail": "  <server>host1</server>",
                "snippet_head": "  <replica>host2</replica>",
            },
            {
                "op": "replace_symbol", "symbol": "cache", "path": "direct_swap",
                "snippet": '  <cache enabled="false">memcached</cache>\n',
                "start_line": 4, "end_line": 4,
                "span_head": '  <cache enabled="true">redis</cache>',
                "span_tail": '  <cache enabled="true">redis</cache>',
                "snippet_head": '  <cache enabled="false">memcached</cache>',
            },
            {
                "op": "delete_symbol", "symbol": "limits",
                "start_line": 5, "end_line": 7,
                "span_head": "  <limits>", "span_tail": "  </limits>",
            },
        ],
    },
    {
        "language": "markdown", "ext": "md", "filename": "original.md",
        "symbol_semantics": (
            "sections (heading + body), named by the heading text; sections "
            "nest so a parent section's replace/delete takes subsections with "
            "it — target leaf sections. The markdown grammar is structurally "
            "error-tolerant (plan §3 risk 1): md validity leans on text "
            "traits, never the parse gate alone"
        ),
        "original": '''# Guide

Intro paragraph.

## Setup

Install steps.

## Usage

Usage notes.
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Setup",
                "snippet": "## Troubleshooting\n\nCommon fixes.\n",
                "anchor_start_line": 5, "anchor_head": "## Setup",
                "anchor_end_line": 8, "anchor_tail": "",
                "snippet_head": "## Troubleshooting",
            },
            {
                "op": "replace_symbol", "symbol": "Usage", "path": "direct_swap",
                "snippet": "## Usage\n\nUpdated usage notes.\n",
                "start_line": 9, "end_line": 11,
                "span_head": "## Usage", "span_tail": "Usage notes.",
                "snippet_head": "## Usage",
            },
            {
                "op": "delete_symbol", "symbol": "Usage",
                "start_line": 9, "end_line": 11,
                "span_head": "## Usage", "span_tail": "Usage notes.",
            },
        ],
        "gigo": {
            "defect": "malformed table (row missing its separator column count) — parses clean by grammar leniency, preserved byte-exact; structural md validity is D1/D3 territory",
            "defect_lines": [3, 4],
            "original": '''# Guide

| broken | table |
| --- |

## Setup

Install steps.
''',
            "op": {
                "op": "insert_after", "symbol": "Setup",
                "snippet": "## Notes\n\nNotes body.\n",
                "anchor_start_line": 6, "anchor_head": "## Setup",
                "anchor_end_line": 8, "anchor_tail": "Install steps.",
                "snippet_head": "## Notes",
            },
        },
    },
    {
        "language": "json", "ext": "json", "filename": "original.json",
        "symbol_semantics": (
            "object pairs (keys), named by the key; nested objects' pairs "
            "become nested symbols. JSON comma semantics constrain insertion "
            "anchors: an inserted pair must carry its own trailing comma and "
            "the anchor pair must already have one (the last pair cannot "
            "anchor an insert)"
        ),
        "original": '''{
  "name": "fastedit",
  "version": "0.5.0",
  "config": {
    "debug": true,
    "level": 3
  },
  "tags": ["ast", "edit"]
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "name",
                "snippet": '  "license": "MIT",\n',
                "anchor_start_line": 2, "anchor_head": '  "name": "fastedit",',
                "anchor_end_line": 2, "anchor_tail": '  "name": "fastedit",',
                "snippet_head": '  "license": "MIT",',
            },
            {
                "op": "delete_symbol", "symbol": "version",
                "start_line": 3, "end_line": 3,
                "span_head": '  "version": "0.5.0",',
                "span_tail": '  "version": "0.5.0",',
            },
        ],
        "unsupported": [
            {
                "op": "replace_symbol",
                "reason": (
                    "no deterministic path can re-define a JSON pair: a bare "
                    "pair is not a standalone-parsable JSON document (the "
                    "direct-swap gate's parse precondition fails), and the "
                    "text-match editor cannot justify a same-key value "
                    "rewrite — JSON's `key: value` separator is not an "
                    "assignment (`=`), so the pair has no replacement key, "
                    "and a marker-free rewrite of a preserved line is "
                    "declined as ambiguous. JSON value updates are served by "
                    "the model path, which is out of the deterministic golden "
                    "matrix's scope"
                ),
            },
        ],
        "gigo": {
            "defect": "missing comma between two members — preserved byte-exact",
            "defect_lines": [2, 3],
            "original": '''{
  "name": "fastedit"
  "version": "0.5.0",
  "config": {
    "debug": true
  }
}
''',
            "op": {
                "op": "insert_after", "symbol": "version",
                "snippet": '  "license": "MIT",\n',
                "anchor_start_line": 3, "anchor_head": '  "version": "0.5.0",',
                "anchor_end_line": 3, "anchor_tail": '  "version": "0.5.0",',
                "snippet_head": '  "license": "MIT",',
            },
        },
    },
    {
        "language": "yaml", "ext": "yaml", "filename": "original.yaml",
        "symbol_semantics": (
            "block-mapping pairs (keys), named by the key (quotes stripped); "
            "nested mappings' pairs become nested symbols"
        ),
        "original": '''name: fastedit
version: 0.5.0
database:
  host: localhost
  port: 5432
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "version",
                "snippet": "license: MIT\n",
                "anchor_start_line": 2, "anchor_head": "version: 0.5.0",
                "anchor_end_line": 2, "anchor_tail": "version: 0.5.0",
                "snippet_head": "license: MIT",
            },
            {
                "op": "replace_symbol", "symbol": "version", "path": "direct_swap",
                "snippet": "version: 0.6.0\n",
                "start_line": 2, "end_line": 2,
                "span_head": "version: 0.5.0", "span_tail": "version: 0.5.0",
                "snippet_head": "version: 0.6.0",
            },
            {
                "op": "delete_symbol", "symbol": "port",
                "start_line": 5, "end_line": 5,
                "span_head": "  port: 5432", "span_tail": "  port: 5432",
            },
        ],
        "gigo": {
            "defect": "unclosed flow sequence in an unrelated key — preserved byte-exact",
            "defect_lines": [5, 6],
            "original": '''name: fastedit
version: 0.5.0
database:
  host: localhost
broken: [unclosed
  - bad
''',
            "op": {
                "op": "insert_after", "symbol": "database",
                "snippet": "cache:\n  enabled: true\n",
                "anchor_start_line": 3, "anchor_head": "database:",
                "anchor_end_line": 4, "anchor_tail": "  host: localhost",
                "snippet_head": "cache:",
            },
        },
    },
    {
        "language": "css", "ext": "css", "filename": "original.css",
        "symbol_semantics": (
            "rule sets, named by their full selector text (`.header`, `body`, "
            "`a:hover`); rules inside @media blocks are found by structural "
            "descent — @media itself is a container, not a symbol"
        ),
        "original": '''body {
  color: #333;
  margin: 0 auto;
}

.header {
  font-weight: bold;
}

.footer {
  padding: 4px;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": ".header",
                "snippet": ".nav {\n  display: flex;\n}\n",
                "anchor_start_line": 6, "anchor_head": ".header {",
                "anchor_end_line": 8, "anchor_tail": "}",
                "snippet_head": ".nav {",
            },
            {
                "op": "replace_symbol", "symbol": ".footer", "path": "direct_swap",
                "snippet": ".footer {\n  padding: 8px;\n}\n",
                "start_line": 10, "end_line": 12,
                "span_head": ".footer {", "span_tail": "}",
                "snippet_head": ".footer {",
            },
            {
                "op": "delete_symbol", "symbol": "body",
                "start_line": 1, "end_line": 4,
                "span_head": "body {", "span_tail": "}",
            },
        ],
        "gigo": {
            "defect": "unclosed rule block (.broken) at EOF — preserved byte-exact",
            "defect_lines": [9, 10],
            "original": '''.header {
  font-weight: bold;
}

.footer {
  padding: 4px;
}

.broken {
  color: red;
''',
            "op": {
                "op": "insert_after", "symbol": ".footer",
                "snippet": ".nav {\n  display: flex;\n}\n",
                "anchor_start_line": 5, "anchor_head": ".footer {",
                "anchor_end_line": 7, "anchor_tail": "}",
                "snippet_head": ".nav {",
            },
        },
    },
    {
        "language": "bash", "ext": "sh", "filename": "original.sh",
        "symbol_semantics": (
            "function definitions (the `name` field is conventional). Shell "
            "variable assignments are deliberately not symbols"
        ),
        "original": '''#!/usr/bin/env bash
VERSION="1.2.3"

run_build() {
  echo "building ${VERSION}"
}

run_test() {
  echo "testing"
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "run_build",
                "snippet": 'run_clean() {\n  echo "cleaning"\n}\n',
                "anchor_start_line": 4, "anchor_head": "run_build() {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": "run_clean() {",
            },
            {
                "op": "replace_symbol", "symbol": "run_test", "path": "direct_swap",
                "snippet": 'run_test() {\n  echo "running tests"\n}\n',
                "start_line": 8, "end_line": 10,
                "span_head": "run_test() {", "span_tail": "}",
                "snippet_head": "run_test() {",
            },
            {
                "op": "delete_symbol", "symbol": "run_build",
                "start_line": 4, "end_line": 6,
                "span_head": "run_build() {", "span_tail": "}",
            },
        ],
    },
    {
        "language": "toml", "ext": "toml", "filename": "original.toml",
        "symbol_semantics": (
            "tables and table-array elements, named by the (possibly dotted) "
            "key text; top-level bare pairs are deliberately not symbols — "
            "the table is the TOML addressing unit"
        ),
        "original": '''title = "fastedit"

[package]
name = "fastedit"
version = "0.5.0"

[dependencies]
serde = "1"

[tool.pytest]
minversion = "7.0"
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "dependencies",
                "snippet": '[dev-dependencies]\npytest = "7.0"\n',
                "anchor_start_line": 7, "anchor_head": "[dependencies]",
                "anchor_end_line": 9, "anchor_tail": "",
                "snippet_head": "[dev-dependencies]",
            },
            {
                "op": "replace_symbol", "symbol": "tool.pytest", "path": "direct_swap",
                "snippet": '[tool.pytest]\nminversion = "8.0"\n',
                "start_line": 10, "end_line": 11,
                "span_head": "[tool.pytest]", "span_tail": 'minversion = "7.0"',
                "snippet_head": "[tool.pytest]",
            },
            {
                "op": "delete_symbol", "symbol": "package",
                "start_line": 3, "end_line": 6,
                "span_head": "[package]", "span_tail": "",
            },
        ],
    },
    {
        "language": "sql", "ext": "sql", "filename": "original.sql",
        "symbol_semantics": (
            "CREATE statements that define a schema object (table / view / "
            "materialized view / index), named by the object name; DML "
            "(SELECT/INSERT/UPDATE) is an operation, never a symbol"
        ),
        "original": '''CREATE TABLE users (
  id INT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT
);

CREATE INDEX idx_orders_user ON orders (user_id);
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "users",
                "snippet": "CREATE TABLE sessions (\n  id INT PRIMARY KEY\n);\n",
                "anchor_start_line": 1, "anchor_head": "CREATE TABLE users (",
                "anchor_end_line": 4, "anchor_tail": ");",
                "snippet_head": "CREATE TABLE sessions (",
            },
            {
                "op": "replace_symbol", "symbol": "orders", "path": "direct_swap",
                "snippet": "CREATE TABLE orders (\n  id INT PRIMARY KEY,\n  user_id INT NOT NULL\n);\n",
                "start_line": 6, "end_line": 9,
                "span_head": "CREATE TABLE orders (", "span_tail": ");",
                "snippet_head": "CREATE TABLE orders (",
            },
            {
                "op": "delete_symbol", "symbol": "idx_orders_user",
                "start_line": 11, "end_line": 11,
                "span_head": "CREATE INDEX idx_orders_user ON orders (user_id);",
                "span_tail": "CREATE INDEX idx_orders_user ON orders (user_id);",
            },
        ],
        "gigo": {
            "defect": "broken CREATE TABLE (missing closing paren) in an unrelated statement — preserved byte-exact",
            "defect_lines": [7, 9],
            "original": '''CREATE TABLE users (
  id INT PRIMARY KEY
);

CREATE INDEX idx_users ON users (id);

CREATE TABLE broken (
  id INT
;
''',
            "op": {
                "op": "insert_after", "symbol": "users",
                "snippet": "CREATE TABLE sessions (\n  id INT PRIMARY KEY\n);\n",
                "anchor_start_line": 1, "anchor_head": "CREATE TABLE users (",
                "anchor_end_line": 3, "anchor_tail": ");",
                "snippet_head": "CREATE TABLE sessions (",
            },
        },
    },
    {
        "language": "dockerfile", "ext": "dockerfile", "filename": "original.dockerfile",
        "symbol_semantics": (
            "FROM instructions (build stages), named by the `AS` alias when "
            "present, else by the image spec; other instructions are steps, "
            "not symbols"
        ),
        "original": '''FROM python:3.12 AS builder

RUN pip install requests

FROM alpine AS runtime

COPY --from=builder /app /app

CMD ["python"]
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "builder",
                "snippet": "WORKDIR /build\n",
                "anchor_start_line": 1, "anchor_head": "FROM python:3.12 AS builder",
                "anchor_end_line": 1, "anchor_tail": "FROM python:3.12 AS builder",
                "snippet_head": "WORKDIR /build",
            },
            {
                "op": "replace_symbol", "symbol": "runtime", "path": "direct_swap",
                "snippet": "FROM alpine:3.19 AS runtime\n",
                "start_line": 5, "end_line": 5,
                "span_head": "FROM alpine AS runtime",
                "span_tail": "FROM alpine AS runtime",
                "snippet_head": "FROM alpine:3.19 AS runtime",
            },
            {
                "op": "delete_symbol", "symbol": "builder",
                "start_line": 1, "end_line": 1,
                "span_head": "FROM python:3.12 AS builder",
                "span_tail": "FROM python:3.12 AS builder",
            },
        ],
    },
    # --- all-grammars extra samples (one representative op each) ---------
    {
        "language": "lua", "ext": "lua", "filename": "original.lua",
        "requires_wheel": "tree_sitter_lua",
        "symbol_semantics": (
            "function declarations, dotted names verbatim (`M.alpha`). "
            "all-grammars extra: extension-unwired by design, so the golden "
            "runner passes the language explicitly (the resolver's "
            "`resolve when requested explicitly` contract)"
        ),
        "original": '''local M = {}

function M.alpha(x)
  return x + 1
end

function M.beta(y)
  return y * 2
end
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "M.alpha",
                "snippet": "function M.gamma(z)\n  return z - 1\nend\n",
                "anchor_start_line": 3, "anchor_head": "function M.alpha(x)",
                "anchor_end_line": 5, "anchor_tail": "end",
                "snippet_head": "function M.gamma(z)",
            },
        ],
    },
    {
        "language": "scala", "ext": "scala", "filename": "original.scala",
        "requires_wheel": "tree_sitter_scala",
        "symbol_semantics": "objects/classes/traits, defs (methods inside a type)",
        "original": '''object Sample {
  val limit = 10

  def alpha(x: Int): Int = x + 1

  def beta(y: Int): Int = y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "  def gamma(z: Int): Int = z - 1\n",
                "anchor_start_line": 4,
                "anchor_head": "  def alpha(x: Int): Int = x + 1",
                "anchor_end_line": 4, "anchor_tail": "  def alpha(x: Int): Int = x + 1",
                "snippet_head": "  def gamma(z: Int): Int = z - 1",
            },
        ],
    },
    {
        "language": "graphql", "ext": "graphql", "filename": "original.graphql",
        "requires_wheel": "tree_sitter_graphql",
        "symbol_semantics": "schema type definitions (object/interface/enum/union/input/scalar), named by their `name` field",
        "original": '''type User {
  id: ID!
  name: String
}

type Post {
  id: ID!
  author: User
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "User",
                "snippet": "type Comment {\n  id: ID!\n}\n",
                "anchor_start_line": 1, "anchor_head": "type User {",
                "anchor_end_line": 4, "anchor_tail": "}",
                "snippet_head": "type Comment {",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Verification + writers
# ---------------------------------------------------------------------------

def _verify_op(original_text: str, op: dict, *, gigo: bool = False) -> None:
    """Cross-check a hand-declared op against the authored original.

    Raises AssertionError when a declared line index does not match the
    fixture it claims to describe — the oracle is only as honest as these
    checks.
    """
    lines = original_text.splitlines(keepends=True)
    op_kind = op["op"]
    where = f"{op_kind} {op['symbol']}" + (" (gigo)" if gigo else "")
    if op_kind == "insert_after":
        anchor_end = op["anchor_end_line"]
        assert 1 <= anchor_end <= len(lines), f"{where}: anchor out of range"
        tail = lines[anchor_end - 1].rstrip("\n")
        assert tail == op["anchor_tail"], (
            f"{where}: anchor line {anchor_end} is {tail!r}, "
            f"expected {op['anchor_tail']!r}"
        )
        # The pipeline aligns the snippet to the anchor's FIRST line, so the
        # pre-alignment precondition compares against that line's indent.
        anchor_head = lines[op["anchor_start_line"] - 1].rstrip("\n")
        assert anchor_head.startswith(op["anchor_head"]), (
            f"{where}: anchor head line {op['anchor_start_line']} is "
            f"{anchor_head!r}, expected to start with {op['anchor_head']!r}"
        )
        snippet_base = op.get("snippet_head", "")
        target_base = anchor_head
    else:
        start, end = op["start_line"], op["end_line"]
        assert 1 <= start <= end <= len(lines), f"{where}: span out of range"
        head = lines[start - 1].rstrip("\n")
        assert head.startswith(op["span_head"]), (
            f"{where}: line {start} is {head!r}, expected to start with "
            f"{op['span_head']!r}"
        )
        tail = lines[end - 1].rstrip("\n")
        assert tail == op["span_tail"], (
            f"{where}: span end line {end} is {tail!r}, "
            f"expected {op['span_tail']!r}"
        )
        snippet_base = op.get("snippet_head", "")
        target_base = head
    if snippet_base and not op.get("prepend_signature_lines"):
        # Pre-alignment precondition: the pipeline's indent alignment is a
        # no-op on every golden op by construction. Ops that declare a
        # prepended signature are exempt — the pipeline pins the span's own
        # (indented) signature lines ahead of the body-only snippet.
        assert _indent_of(snippet_base) == _indent_of(target_base), (
            f"{where}: snippet indent {_indent_of(snippet_base)!r} does not "
            f"match target indent {_indent_of(target_base)!r}"
        )


def _expected_bytes(original_text: str, op: dict) -> bytes:
    """The oracle's expected output for one op (pure line arithmetic)."""
    lines = original_text.splitlines(keepends=True)
    op_kind = op["op"]
    if op_kind == "insert_after":
        merged = oracle_insert_after(lines, op["anchor_end_line"], op["snippet"])
    elif op_kind == "replace_symbol":
        merged = oracle_replace_span(
            lines, op["start_line"], op["end_line"], op["snippet"],
            op.get("prepend_signature_lines", 0),
        )
    elif op_kind == "delete_symbol":
        merged = oracle_delete_span(lines, op["start_line"], op["end_line"])
    else:
        raise AssertionError(f"unknown op kind {op_kind!r}")
    return "".join(merged).encode("utf-8")


def _op_manifest(op: dict) -> dict:
    return {
        "op": op["op"],
        "symbol": op["symbol"],
        "snippet": op.get("snippet"),
        "path": op.get("path") or (
            "fast_path" if op["op"] == "insert_after" else None
        ),
        "prepend_signature_lines": op.get("prepend_signature_lines", 0),
        "oracle": {
            key: op[key]
            for key in ("anchor_start_line", "anchor_end_line", "start_line", "end_line")
            if key in op
        },
    }


def _write_lang(entry: dict) -> None:
    lang = entry["language"]
    lang_dir = GOLDEN_DIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    ext = entry["ext"]
    filename = entry["filename"]
    original_text = entry["original"]

    manifest: dict = {
        "language": lang,
        "ext": ext,
        "filename": filename,
        "symbol_semantics": entry["symbol_semantics"],
        "ops": [],
    }
    if entry.get("requires_wheel"):
        manifest["requires_wheel"] = entry["requires_wheel"]

    (lang_dir / filename).write_bytes(original_text.encode("utf-8"))

    for op in entry["ops"]:
        _verify_op(original_text, op)
        expected_name = f"expected_{op['op']}_{_sanitize(op['symbol'])}.{ext}"
        (lang_dir / expected_name).write_bytes(_expected_bytes(original_text, op))
        manifest["ops"].append({
            **_op_manifest(op),
            "expected": expected_name,
        })
    if entry.get("unsupported"):
        manifest["unsupported"] = entry["unsupported"]

    if entry.get("gigo"):
        gigo = entry["gigo"]
        gigo_original = gigo["original"]
        (lang_dir / f"original_gigo.{ext}").write_bytes(
            gigo_original.encode("utf-8"),
        )
        _verify_op(gigo_original, gigo["op"], gigo=True)
        expected_name = f"expected_gigo_{_sanitize(gigo['op']['symbol'])}.{ext}"
        (lang_dir / expected_name).write_bytes(
            _expected_bytes(gigo_original, gigo["op"]),
        )
        manifest["gigo"] = {
            "original": f"original_gigo.{ext}",
            "defect": gigo["defect"],
            "defect_lines": gigo["defect_lines"],
            "op": {**_op_manifest(gigo["op"]), "expected": expected_name},
        }

    (lang_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for entry in LANGUAGES:
        _write_lang(entry)
        ops = len(entry["ops"]) + (1 if entry.get("gigo") else 0)
        print(f"{entry['language']}: {ops} golden op(s) written")


if __name__ == "__main__":
    main()
