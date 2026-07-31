# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for check_license_header.py module."""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_license_header import (
    COPYRIGHT_LINE_TEMPLATE,
    LICENSE_HEADER,
    LICENSE_LINE,
    apply_fixes,
    build_excludes,
    check_file_header,
    classify_files,
    create_parser,
    extract_content_after_preamble,
    find_python_files,
    fix_file_header,
    get_current_year,
    get_files_to_check,
    has_copyright_header,
    has_license_header,
    main,
    normalize_header,
    parse_gitignore,
    report_issues,
    should_exclude,
)

OLD_DFFL_HEADER = (
    "# DeepFellow Software Framework.\n"
    "# Copyright © 2020 Simplito sp. z o.o.\n"
    "#\n"
    "# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).\n"
    "# This software is Licensed under the DeepFellow Free License.\n"
    "#\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
)


@pytest.fixture
def valid_content() -> str:
    """Valid Python file content with license header."""
    return f"{LICENSE_HEADER}\n\nimport sys\n"


@pytest.fixture
def valid_with_shebang() -> str:
    """Valid Python file with shebang and license header."""
    return f"#!/usr/bin/env python3\n\n{LICENSE_HEADER}\n\nimport sys\n"


@pytest.fixture
def valid_with_encoding() -> str:
    """Valid Python file with encoding declaration and license header."""
    return f"# -*- coding: utf-8 -*-\n\n{LICENSE_HEADER}\n\nimport sys\n"


@pytest.fixture
def valid_with_both() -> str:
    """Valid Python file with shebang, encoding, and license header."""
    return f"#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n{LICENSE_HEADER}\n\nimport sys\n"


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Sample project structure for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "src" / "utils.py").write_text("# utils")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "main.cpython-311.pyc").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("# test")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("# venv")
    return tmp_path


@pytest.fixture
def gitignore_path(tmp_path: Path):
    """Fixture to write content to a .gitignore file in a temp directory."""
    return tmp_path / ".gitignore"


def test_has_license_header_present() -> None:
    assert has_license_header(f"{LICENSE_HEADER}\nimport sys\n") is True


def test_has_license_header_absent() -> None:
    assert has_license_header("import sys\n") is False


def test_has_license_header_old_dffl_header_not_recognized() -> None:
    assert has_license_header(f"{OLD_DFFL_HEADER}\nimport sys\n") is False


def test_has_copyright_header_present() -> None:
    assert has_copyright_header(f"{COPYRIGHT_LINE_TEMPLATE.format(year=2024)}\nimport sys\n") is True


def test_has_copyright_header_absent() -> None:
    assert has_copyright_header("import sys\n") is False


def test_has_copyright_header_any_year_accepted() -> None:
    """The copyright year is never re-validated once set - any 4-digit year matches."""
    assert has_copyright_header(f"{COPYRIGHT_LINE_TEMPLATE.format(year=1999)}\n") is True


@pytest.mark.parametrize(
    ("input_str", "expected_output"),
    [
        ("line1   \nline2  ", "line1\nline2"),
        ("\n\n  hello  \n\n", "hello"),
        ("line1\n\nline3", "line1\n\nline3"),
    ],
    ids=["trailing_whitespace", "outer_whitespace", "inner_structure"],
)
def test_normalize_header(input_str: str, expected_output: str) -> None:
    assert normalize_header(input_str) == expected_output


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("# License\nimport sys", "# License\nimport sys"),
        ("#!/usr/bin/env python3\n\n# License", "# License"),
        ("# -*- coding: utf-8 -*-\n\n# License", "# License"),
        ("#!/usr/bin/env python3\n# coding: utf-8\n\n# License", "# License"),
    ],
    ids=["no_preamble", "with_shebang", "with_encoding", "with_shebang_and_encoding"],
)
def test_extract_content_after_preamble(content: str, expected: str) -> None:
    assert extract_content_after_preamble(content) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("__pycache__\n.venv\n*.pyc\nbuild/\n", {"__pycache__", ".venv", "*.pyc", "build"}),
        ("# This is a comment\n__pycache__\n# Another comment\n.venv\n", {"__pycache__", ".venv"}),
        ("__pycache__\n\n\n.venv\n", {"__pycache__", ".venv"}),
        ("/build/\n/dist\nvenv/\n", {"build", "dist", "venv"}),
        ("# Python\n__pycache__/\n*.py[cod]\n*.egg-info/\n\n# IDE\n.idea/", {"__pycache__", "*.py[cod]", "*.egg-info", ".idea"}),
    ],
    ids=["standard", "comments", "empty_lines", "slashes", "complex"],
)
def test_parse_gitignore_logic(gitignore_path: Path, tmp_path: Path, content: str, expected: str):
    gitignore_path.write_text(content)

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset(expected)


def test_parse_gitignore_returns_empty_if_no_gitignore(tmp_path: Path):
    """Separate test for the missing file case as it doesn't need the fixture."""
    assert parse_gitignore(tmp_path) == frozenset()


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        (Path("src/__pycache__/module.pyc"), "__pycache__", True),
        (Path("project/.venv/lib/site.py"), ".venv", True),
        (Path("pkg.egg-info/PKG-INFO"), "*.egg-info", True),
        (Path("src/mymodule/main.py"), "__pycache__", False),
        (Path("tests/test_main.py"), ".venv", False),
        (Path("src/cache.pyc"), "*.pyc", True),
        (Path("tests/test_utils.py"), "test_*", True),
    ],
)
def test_should_exclude_patterns(path: Path, pattern: str, expected: bool) -> None:
    assert should_exclude(path, frozenset({pattern})) is expected


@pytest.mark.parametrize(
    ("path", "excludes", "expected"),
    [
        (Path("src/__pycache__/cached.pyc"), frozenset({"__pycache__", ".venv"}), True),
        (Path("src/main.py"), frozenset(), False),
    ],
    ids=["multiple_excludes", "no_excludes"],
)
def test_should_exclude(path: Path, excludes: frozenset[str], expected: bool) -> None:
    assert should_exclude(path, excludes) is expected


def test_check_file_header_valid(tmp_path: Path, valid_content: str) -> None:
    file = tmp_path / "valid.py"
    file.write_text(valid_content)

    assert check_file_header(file) is True


def test_check_file_header_valid_with_shebang(tmp_path: Path, valid_with_shebang: str) -> None:
    file = tmp_path / "with_shebang.py"
    file.write_text(valid_with_shebang)

    assert check_file_header(file) is True


def test_check_file_header_valid_with_encoding(tmp_path: Path, valid_with_encoding: str) -> None:
    file = tmp_path / "with_encoding.py"
    file.write_text(valid_with_encoding)

    assert check_file_header(file) is True


def test_check_file_header_valid_with_shebang_and_encoding(tmp_path: Path, valid_with_both: str) -> None:
    file = tmp_path / "with_both.py"
    file.write_text(valid_with_both)

    assert check_file_header(file) is True


def test_check_file_header_invalid_no_header(tmp_path: Path) -> None:
    file = tmp_path / "no_header.py"
    file.write_text("import sys\n")

    assert check_file_header(file) is False


def test_check_file_header_invalid_wrong_header(tmp_path: Path) -> None:
    file = tmp_path / "wrong_header.py"
    file.write_text("# MIT License\nimport sys\n")

    assert check_file_header(file) is False


def test_check_file_header_invalid_old_dffl_header(tmp_path: Path) -> None:
    """A file still carrying the old DFFL header is not considered valid."""
    file = tmp_path / "old_dffl.py"
    file.write_text(f"{OLD_DFFL_HEADER}\nimport sys\n")

    assert check_file_header(file) is False


def test_check_file_header_missing_copyright_only(tmp_path: Path) -> None:
    """License line alone is not enough - the copyright line is also required."""
    file = tmp_path / "license_only.py"
    file.write_text(f"{LICENSE_LINE}\nimport sys\n")

    assert check_file_header(file) is False


def test_check_file_header_missing_license_only(tmp_path: Path) -> None:
    """Copyright line alone is not enough - the license line is also required."""
    file = tmp_path / "copyright_only.py"
    file.write_text(f"{COPYRIGHT_LINE_TEMPLATE.format(year=2024)}\nimport sys\n")

    assert check_file_header(file) is False


def test_check_file_header_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.py"
    file.write_text("")

    assert check_file_header(file) is True


def test_check_file_header_whitespace_only(tmp_path: Path) -> None:
    file = tmp_path / "whitespace.py"
    file.write_text("   \n\n")

    assert check_file_header(file) is True


def test_check_file_header_nonexistent_file(tmp_path: Path) -> None:
    assert check_file_header(tmp_path / "nonexistent.py") is False


def test_fix_file_header_without_header(tmp_path: Path) -> None:
    file = tmp_path / "no_header.py"
    file.write_text("import sys\n\ndef main():\n    pass\n")

    result = fix_file_header(file)

    assert result is True
    assert check_file_header(file) is True
    content = file.read_text()
    assert "# SPDX-License-Identifier: MIT" in content
    assert "import sys" in content


def test_fix_file_header_with_shebang(tmp_path: Path) -> None:
    file = tmp_path / "with_shebang.py"
    file.write_text("#!/usr/bin/env python3\n\nimport sys\n")

    fix_file_header(file)

    assert check_file_header(file) is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n")
    assert "# SPDX-License-Identifier: MIT" in content
    assert "import sys" in content


def test_fix_file_header_with_encoding(tmp_path: Path) -> None:
    file = tmp_path / "with_encoding.py"
    file.write_text("# -*- coding: utf-8 -*-\n\nimport sys\n")

    fix_file_header(file)

    assert check_file_header(file) is True
    content = file.read_text()
    assert content.startswith("# -*- coding: utf-8 -*-\n")
    assert "# SPDX-License-Identifier: MIT" in content


def test_fix_file_header_with_shebang_and_encoding(tmp_path: Path) -> None:
    file = tmp_path / "with_both.py"
    file.write_text("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nimport sys\n")

    fix_file_header(file)

    assert check_file_header(file) is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n")
    assert "# SPDX-License-Identifier: MIT" in content


def test_fix_file_header_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.py"
    file.write_text("")

    fix_file_header(file)

    content = file.read_text()
    assert "# SPDX-License-Identifier: MIT" in content


def test_fix_file_header_migrates_old_dffl_header(tmp_path: Path) -> None:
    """Fix should strip the old DFFL header and add the SPDX header instead."""
    file = tmp_path / "old_dffl.py"
    file.write_text(f"{OLD_DFFL_HEADER}\nimport sys\n")

    result = fix_file_header(file)

    assert result is True
    assert check_file_header(file) is True
    content = file.read_text()
    assert "# SPDX-License-Identifier: MIT" in content
    assert "DeepFellow Software Framework" not in content
    assert "import sys" in content


def test_fix_file_header_adds_missing_copyright_only(tmp_path: Path) -> None:
    """A file with only the license line gets just the copyright line appended."""
    file = tmp_path / "license_only.py"
    file.write_text(f"{LICENSE_LINE}\nimport sys\n")

    result = fix_file_header(file)

    assert result is True
    assert check_file_header(file) is True
    content = file.read_text()
    assert content.count("SPDX-License-Identifier") == 1
    assert "SPDX-FileCopyrightText" in content


def test_fix_file_header_adds_missing_license_only(tmp_path: Path) -> None:
    """A file with only the copyright line gets just the license line appended."""
    file = tmp_path / "copyright_only.py"
    file.write_text(f"{COPYRIGHT_LINE_TEMPLATE.format(year=2024)}\nimport sys\n")

    result = fix_file_header(file)

    assert result is True
    assert check_file_header(file) is True
    content = file.read_text()
    assert "SPDX-License-Identifier: MIT" in content
    assert content.count("SPDX-FileCopyrightText") == 1


def test_fix_file_header_copyright_only_no_match_returns_false(tmp_path: Path) -> None:
    """Defensive guard: if the copyright pattern search unexpectedly returns None, fix_file_header bails out."""
    file = tmp_path / "copyright_only.py"
    file.write_text(f"{COPYRIGHT_LINE_TEMPLATE.format(year=2024)}\nimport sys\n")

    with patch("scripts.check_license_header.COPYRIGHT_HEADER_PATTERN") as mock_pattern:
        mock_pattern.search.side_effect = [object(), None]

        result = fix_file_header(file)

    assert result is False


def test_fix_file_header_pins_year_never_bumped(tmp_path: Path) -> None:
    """An existing copyright year is left untouched by --fix, even if it's stale."""
    file = tmp_path / "stale_year.py"
    file.write_text(f"{LICENSE_LINE}\n{COPYRIGHT_LINE_TEMPLATE.format(year=2020)}\nimport sys\n")

    fix_file_header(file)

    content = file.read_text()
    assert "SPDX-FileCopyrightText: 2020 Simplito sp. z o.o." in content
    assert f"SPDX-FileCopyrightText: {get_current_year()} Simplito sp. z o.o." not in content


def test_fix_file_header_preserves_code(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    file.write_text("import sys\n\ndef main():\n    print('hello')\n")

    fix_file_header(file)

    content = file.read_text()
    assert "import sys" in content
    assert "def main():" in content
    assert "print('hello')" in content


def test_fix_file_header_nonexistent_file(tmp_path: Path) -> None:
    assert fix_file_header(tmp_path / "nonexistent.py") is False


def test_fix_file_header_idempotent(tmp_path: Path, valid_content: str) -> None:
    file = tmp_path / "already_valid.py"
    file.write_text(valid_content)

    assert check_file_header(file) is True

    fix_file_header(file)

    assert check_file_header(file) is True


def test_find_python_files_finds_all(project_dir: Path) -> None:
    excludes = frozenset({"__pycache__", ".venv"})

    files = find_python_files([project_dir], excludes)

    assert {f.name for f in files} == {"main.py", "utils.py", "test_main.py"}


def test_find_python_files_excludes_pycache(project_dir: Path) -> None:
    excludes = frozenset({"__pycache__"})

    files = find_python_files([project_dir], excludes)

    assert not any("__pycache__" in str(f) for f in files)


def test_find_python_files_excludes_venv(project_dir: Path) -> None:
    excludes = frozenset({".venv"})

    files = find_python_files([project_dir], excludes)

    assert not any(".venv" in str(f) for f in files)


def test_find_python_files_non_recursive(project_dir: Path) -> None:
    (project_dir / "root.py").write_text("# root")

    files = find_python_files([project_dir], frozenset(), recursive=False)

    assert len(files) == 1
    assert files[0].name == "root.py"


def test_find_python_files_specific_file(project_dir: Path) -> None:
    target = project_dir / "src" / "main.py"

    files = find_python_files([target], frozenset())

    assert files == [target]


def test_find_python_files_empty_directory(tmp_path: Path) -> None:
    assert find_python_files([tmp_path], frozenset()) == []


def test_find_python_files_respects_gitignore(project_dir: Path) -> None:
    gitignore = project_dir / ".gitignore"
    gitignore.write_text("__pycache__\n.venv\n")
    excludes = parse_gitignore(project_dir)

    files = find_python_files([project_dir], excludes)

    assert {f.name for f in files} == {"main.py", "utils.py", "test_main.py"}


def test_fix_file_header_no_trailing_newline(tmp_path: Path) -> None:
    file = tmp_path / "no_newline.py"
    file.write_text("import sys")

    result = fix_file_header(file)

    assert result is True
    content = file.read_text()
    assert content.endswith("\n")
    assert "# SPDX-License-Identifier: MIT" in content
    assert "import sys" in content


def test_fix_file_header_only_preamble_lines(tmp_path: Path) -> None:
    file = tmp_path / "only_shebang.py"
    file.write_text("#!/usr/bin/env python3\n")

    result = fix_file_header(file)

    assert result is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n")
    assert "# SPDX-License-Identifier: MIT" in content


def test_fix_file_header_preamble_only_no_trailing_newline(tmp_path: Path) -> None:
    """All lines are preamble (no trailing newline), so the for loop exhausts without break."""
    file = tmp_path / "preamble_only.py"
    file.write_text("#!/usr/bin/env python3\n# coding: utf-8")

    result = fix_file_header(file)

    assert result is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n# coding: utf-8\n")
    assert "# SPDX-License-Identifier: MIT" in content


def test_parse_gitignore_skips_slash_only_lines(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("/\n__pycache__\n")

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset({"__pycache__"})


def test_parse_gitignore_handles_oserror(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__\n")

    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset()


def test_extract_content_after_preamble_all_lines_are_preamble() -> None:
    content = "#!/usr/bin/env python3\n# coding: utf-8"
    result = extract_content_after_preamble(content)
    assert result == ""


def test_should_exclude_wildcard_prefix_no_match() -> None:
    assert should_exclude(Path("src/main.py"), frozenset({"*.pyc"})) is False


def test_should_exclude_wildcard_prefix_matches_part() -> None:
    assert should_exclude(Path("pkg.egg-info/PKG-INFO"), frozenset({"*.egg-info"})) is True


def test_find_python_files_non_py_file(tmp_path: Path) -> None:
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("hello")

    files = find_python_files([txt_file], frozenset())

    assert files == []


def test_find_python_files_nonexistent_path(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent"

    files = find_python_files([nonexistent], frozenset())

    assert files == []


def test_build_excludes_without_gitignore(tmp_path: Path) -> None:
    excludes = build_excludes([tmp_path], [], use_gitignore=False)

    assert ".git" in excludes
    assert ".uv-cache" in excludes


def test_build_excludes_with_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")
    excludes = build_excludes([tmp_path], [], use_gitignore=True)

    assert ".venv" in excludes
    assert "__pycache__" in excludes


def test_build_excludes_with_extra_excludes(tmp_path: Path) -> None:
    excludes = build_excludes([tmp_path], ["custom_dir"], use_gitignore=False)

    assert "custom_dir" in excludes


def test_build_excludes_with_file_path(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".venv\n")
    file_path = tmp_path / "main.py"
    file_path.write_text("")

    excludes = build_excludes([file_path], [], use_gitignore=True)

    assert ".venv" in excludes


def test_get_files_to_check_with_files_list(tmp_path: Path) -> None:
    py_file = tmp_path / "main.py"
    py_file.write_text("")
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("")

    result = get_files_to_check([], [py_file, txt_file], frozenset(), recursive=True)

    assert result == [py_file]


def test_get_files_to_check_with_excluded_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    py_file = cache_dir / "cached.py"
    py_file.write_text("")

    result = get_files_to_check([], [py_file], frozenset({"__pycache__"}), recursive=True)

    assert result == []


def test_get_files_to_check_without_files(project_dir: Path) -> None:
    result = get_files_to_check([project_dir], None, frozenset({"__pycache__", ".venv"}), recursive=True)

    assert len(result) > 0


def test_classify_files_missing_header(tmp_path: Path) -> None:
    f = tmp_path / "no_header.py"
    f.write_text("import sys\n")

    missing = classify_files([f])

    assert f in missing


def test_classify_files_old_dffl_header_counts_as_missing(tmp_path: Path) -> None:
    f = tmp_path / "old_dffl.py"
    f.write_text(f"{OLD_DFFL_HEADER}\nimport sys\n")

    missing = classify_files([f])

    assert f in missing


def test_classify_files_valid(tmp_path: Path, valid_content: str) -> None:
    f = tmp_path / "valid.py"
    f.write_text(valid_content)

    missing = classify_files([f])

    assert f not in missing


def test_classify_files_empty() -> None:
    assert classify_files([]) == []


def test_apply_fixes_successful(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "no_header.py"
    f.write_text("import sys\n")

    apply_fixes([f])

    assert "Fixed:" in capsys.readouterr().out


def test_apply_fixes_failed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "nonexistent.py"

    apply_fixes([f])

    assert "Error:" in capsys.readouterr().out


def test_report_issues_no_issues(capsys: pytest.CaptureFixture[str]) -> None:
    report_issues([], verbose=False)

    assert capsys.readouterr().out == ""


def test_report_issues_missing_non_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "missing.py"

    report_issues([f], verbose=False)

    out = capsys.readouterr().out
    assert "Missing license header in 1 files" in out
    assert "Use --verbose" in out


def test_report_issues_missing_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "missing.py"

    report_issues([f], verbose=True)

    assert str(f) in capsys.readouterr().out


def test_create_parser_defaults() -> None:
    parser = create_parser()

    args = parser.parse_args([])

    assert args.fix is False
    assert args.verbose is False
    assert args.no_recursive is False
    assert args.no_gitignore is False
    assert args.exclude == []
    assert args.files is None


def test_create_parser_all_flags() -> None:
    parser = create_parser()

    args = parser.parse_args(["--fix", "--verbose", "--no-recursive", "--no-gitignore"])

    assert args.fix is True
    assert args.verbose is True
    assert args.no_recursive is True
    assert args.no_gitignore is True


def test_create_parser_paths() -> None:
    parser = create_parser()

    args = parser.parse_args(["src/", "tests/"])

    assert len(args.paths) == 2


def test_create_parser_exclude() -> None:
    parser = create_parser()

    args = parser.parse_args(["--exclude", ".venv", "__pycache__"])

    assert ".venv" in args.exclude
    assert "__pycache__" in args.exclude


def test_create_parser_files() -> None:
    parser = create_parser()

    args = parser.parse_args(["--files", "a.py", "b.py"])

    assert len(args.files) == 2


def _make_args(**kwargs: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "paths": [],
        "files": None,
        "exclude": [],
        "no_recursive": False,
        "no_gitignore": True,
        "fix": False,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_main_all_valid(tmp_path: Path, valid_content: str, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "valid.py").write_text(valid_content)

    result = main(_make_args(paths=[tmp_path]))

    assert result == 0
    assert "valid license headers" in capsys.readouterr().out


def test_main_missing_headers(tmp_path: Path) -> None:
    (tmp_path / "no_header.py").write_text("import sys\n")

    result = main(_make_args(paths=[tmp_path]))

    assert result == 1


def test_main_fix_mode(tmp_path: Path) -> None:
    (tmp_path / "no_header.py").write_text("import sys\n")

    result = main(_make_args(paths=[tmp_path], fix=True))

    assert result == 0


def test_main_fix_mode_no_issues(tmp_path: Path, valid_content: str, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "valid.py").write_text(valid_content)

    result = main(_make_args(paths=[tmp_path], fix=True))

    assert result == 0
    assert "valid license headers" in capsys.readouterr().out


def test_main_verbose_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "no_header.py"
    f.write_text("import sys\n")

    result = main(_make_args(paths=[tmp_path], verbose=True))

    assert result == 1
    assert str(f) in capsys.readouterr().out


def test_main_parses_argv(tmp_path: Path, valid_content: str) -> None:
    (tmp_path / "valid.py").write_text(valid_content)

    with patch("sys.argv", ["check_license_header.py", str(tmp_path), "--no-gitignore"]):
        result = main()

    assert result == 0
