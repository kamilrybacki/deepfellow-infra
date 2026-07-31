#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""License header validation script for Python files."""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# ruff: noqa: T201

LICENSE_LINE = "# SPDX-License-Identifier: MIT"
COPYRIGHT_LINE_TEMPLATE = "# SPDX-FileCopyrightText: {year} Simplito sp. z o.o."


def get_current_year() -> int:
    """Get the current year, used to stamp a new copyright line (never used to re-validate an existing one)."""
    return datetime.now(UTC).year


LICENSE_HEADER = f"{LICENSE_LINE}\n{COPYRIGHT_LINE_TEMPLATE.format(year=get_current_year())}\n"

# Pattern to match the license identifier line
LICENSE_HEADER_PATTERN = re.compile(r"# SPDX-License-Identifier: MIT\s*$", re.MULTILINE)

# Pattern to match the copyright line with any year - once set, the year is never checked or bumped
COPYRIGHT_HEADER_PATTERN = re.compile(r"# SPDX-FileCopyrightText: \d{4} Simplito sp\. z o\.o\.\s*$", re.MULTILINE)

# Pattern to match the old DFFL prose header, so `--fix` can migrate files to the SPDX header
OLD_LICENSE_HEADER_PATTERN = re.compile(
    r"# DeepFellow Software Framework\.\s*\n"
    r"# Copyright © \d{4} Simplito sp\. z o\.o\.\s*\n"
    r"#\s*\n"
    r"# This file is part of the DeepFellow Software Framework \(https://deepfellow\.ai\)\.\s*\n"
    r"# This software is Licensed under the DeepFellow Free License\.\s*\n"
    r"#\s*\n"
    r"# See the License for the specific language governing permissions and\s*\n"
    r"# limitations under the License\.\n*"
)

DEFAULT_EXCLUDES = frozenset({".git", ".uv-cache"})


def parse_gitignore(path: Path) -> frozenset[str]:
    """Parse .gitignore file and return patterns as a frozenset."""
    gitignore = path / ".gitignore"
    if not gitignore.exists():
        return frozenset()

    patterns: set[str] = set()
    try:
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Normalize: remove leading/trailing slashes for directory matching
                pattern = line.strip("/")
                if pattern:
                    patterns.add(pattern)
    except OSError:
        return frozenset()

    return frozenset(patterns)


def normalize_header(text: str) -> str:
    """Normalize header text by stripping trailing whitespace."""
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


def extract_content_after_preamble(content: str) -> str:
    """Extract file content after optional shebang/encoding lines."""
    lines = content.split("\n")
    start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#!") or "coding" in stripped[:20]:
            start_idx = i + 1
            if start_idx < len(lines) and not lines[start_idx].strip():
                start_idx += 1
        else:
            break

    return "\n".join(lines[start_idx:])


def has_license_header(content: str) -> bool:
    """Check whether the SPDX-License-Identifier line is present in the file content."""
    remaining = extract_content_after_preamble(content)
    return LICENSE_HEADER_PATTERN.search(remaining) is not None


def has_copyright_header(content: str) -> bool:
    """Check whether the SPDX-FileCopyrightText line is present in the file content."""
    remaining = extract_content_after_preamble(content)
    return COPYRIGHT_HEADER_PATTERN.search(remaining) is not None


def check_file_header(filepath: Path) -> bool:
    """Check if a file contains both required SPDX header lines."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    if not content.strip():
        return True

    return has_license_header(content) and has_copyright_header(content)


def insert_header_lines(content: str, header_lines: list[str]) -> str:
    """Insert header_lines after any shebang/encoding preamble, ahead of the rest of the content."""
    lines = content.split("\n")
    preamble_lines: list[str] = []
    start_idx = 0

    # Extract shebang and encoding lines
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#!") or "coding" in stripped[:20]:
            preamble_lines.append(line)
            start_idx = i + 1
        else:
            break

    rest = "\n".join(lines[start_idx:]).lstrip("\n")
    parts: list[str] = []

    if preamble_lines:
        parts.append("\n".join(preamble_lines))
        parts.append("")

    parts.append("\n".join(header_lines))

    if rest:
        parts.append("")
        parts.append(rest)

    new_content = "\n".join(parts)
    if not new_content.endswith("\n"):
        new_content += "\n"

    return new_content


def fix_file_header(filepath: Path) -> bool:
    """Remove a stale DFFL header if present and add whichever SPDX line(s) are missing."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    content = OLD_LICENSE_HEADER_PATTERN.sub("", content, count=1)

    if not content.strip():
        filepath.write_text(LICENSE_HEADER, encoding="utf-8")
        return True

    has_license = has_license_header(content)
    has_copyright = has_copyright_header(content)

    if has_license and has_copyright:
        filepath.write_text(content, encoding="utf-8")
        return True

    if has_license and not has_copyright:
        # License line already present somewhere - add the copyright line right after it.
        copyright_line = COPYRIGHT_LINE_TEMPLATE.format(year=get_current_year())
        new_content = content.replace(LICENSE_LINE, f"{LICENSE_LINE}\n{copyright_line}", 1)
        filepath.write_text(new_content, encoding="utf-8")
        return True

    if has_copyright and not has_license:
        # Copyright line already present somewhere - add the license line right before it.
        match = COPYRIGHT_HEADER_PATTERN.search(content)
        if match is None:
            return False
        pos = match.start()
        new_content = content[:pos] + LICENSE_LINE + "\n" + content[pos:]
        filepath.write_text(new_content, encoding="utf-8")
        return True

    # Neither line present - insert the full header block after any shebang/encoding preamble.
    header_lines = [LICENSE_LINE, COPYRIGHT_LINE_TEMPLATE.format(year=get_current_year())]
    filepath.write_text(insert_header_lines(content, header_lines), encoding="utf-8")
    return True


def should_exclude(path: Path, excludes: frozenset[str]) -> bool:
    """Check if a path should be excluded from validation."""
    parts = path.parts
    name = path.name

    for pattern in excludes:
        # Exact name match (e.g., __pycache__, .venv)
        if pattern in parts or name == pattern:
            return True
        # Wildcard prefix (e.g., *.egg-info, *.pyc)
        if pattern.startswith("*"):
            suffix = pattern[1:]
            if name.endswith(suffix) or any(p.endswith(suffix) for p in parts):
                return True
        # Wildcard suffix (e.g., test_*)
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return True

    return False


def find_python_files(
    paths: list[Path],
    excludes: frozenset[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    """Find all Python files in the given paths."""
    python_files: list[Path] = []

    for path in paths:
        if path.is_file():
            if path.suffix == ".py" and not should_exclude(path, excludes):
                python_files.append(path)
        elif path.is_dir():
            glob_func = path.rglob if recursive else path.glob
            for py_file in glob_func("*.py"):
                if not should_exclude(py_file, excludes):
                    python_files.append(py_file)

    return sorted(set(python_files))


def build_excludes(
    paths: list[Path],
    extra_excludes: list[str],
    *,
    use_gitignore: bool,
) -> frozenset[str]:
    """Build the set of exclusion patterns from defaults, gitignore, and CLI args."""
    excludes = set(DEFAULT_EXCLUDES)

    if use_gitignore:
        gitignore_dirs = {p if p.is_dir() else p.parent for p in paths}
        for directory in gitignore_dirs:
            excludes.update(parse_gitignore(directory))
        excludes.update(parse_gitignore(Path.cwd()))

    excludes.update(extra_excludes)
    return frozenset(excludes)


def get_files_to_check(
    paths: list[Path],
    files: list[Path] | None,
    excludes: frozenset[str],
    *,
    recursive: bool,
) -> list[Path]:
    """Determine which files to check based on arguments."""
    if files:
        return [f for f in files if f.suffix == ".py" and not should_exclude(f, excludes)]
    return find_python_files(paths, excludes, recursive=recursive)


def classify_files(
    files_to_check: list[Path],
) -> list[Path]:
    """Return the files that are missing the license header."""
    return [f for f in files_to_check if not check_file_header(f)]


def apply_fixes(files: list[Path]) -> None:
    """Apply fixes to all files and print results."""
    for f in files:
        if fix_file_header(f):
            print(f"Fixed: {f}")
        else:
            print(f"Error: {f}")


def report_issues(
    missing_header: list[Path],
    *,
    verbose: bool,
) -> None:
    """Report validation issues to stdout."""
    if not missing_header:
        return

    if verbose:
        for f in missing_header:
            print(f)
    else:
        print(f"Missing license header in {len(missing_header)} files")
        print("Use --verbose to see individual files")
        print("Use --fix to automatically repair")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate license headers in Python files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="Paths to check (default: current directory)",
    )

    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Specific files to check",
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Additional patterns to exclude",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories",
    )

    parser.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Do not read patterns from .gitignore",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Add missing license headers",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show individual file paths",
    )

    return parser


def main(args: argparse.Namespace | None = None) -> int:
    """Validate license headers in Python files.

    Args:
        args: Parsed arguments. If None, parses from sys.argv.

    Returns:
        Exit code (0 for success, 1 for validation errors).
    """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    paths: list[Path] = args.paths if args.paths else [Path.cwd()]
    files: list[Path] | None = args.files
    exclude: list[str] = args.exclude
    no_recursive: bool = args.no_recursive
    no_gitignore: bool = args.no_gitignore
    fix: bool = args.fix
    verbose: bool = args.verbose

    excludes = build_excludes(paths, exclude, use_gitignore=not no_gitignore)
    files_to_check = get_files_to_check(paths, files, excludes, recursive=not no_recursive)
    missing_header = classify_files(files_to_check)

    if fix and missing_header:
        apply_fixes(missing_header)
        return 0

    report_issues(missing_header, verbose=verbose)

    if missing_header:
        return 1

    print("All Python files have valid license headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
