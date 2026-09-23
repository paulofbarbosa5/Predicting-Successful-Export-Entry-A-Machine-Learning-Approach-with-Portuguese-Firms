#!/usr/bin/env python3
"""
find_hyperparams.py

Audit helper for the replication package.

It reports:
1. Python and package versions from the active environment.
2. Text outputs that appear to store selected hyperparameters.
3. Hyperparameter grids, class weights, and random seeds found in Python code.
4. Model-performance CSV files and the validation splits they contain.
5. Candidate metric rows for the random-firm-year and firm-grouped splits.

Safety:
- The script never opens Parquet, RData, DTA, pickle, joblib, or model files.
- It skips confidential or row-level directories by default:
  data/raw, data/processed, outputs/predictions, and outputs/models.
- It reads only small text files.

Recommended location:
    <replication-package>/find_hyperparams.py

Usage:
    python find_hyperparams.py

Optional:
    python find_hyperparams.py --root "C:\\path\\to\\distance-to-export-python"
    python find_hyperparams.py --output outputs/diagnostics/my_audit.txt
    python find_hyperparams.py --no-write
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


# Directories that should never be scanned.
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}

# Confidential or row-level paths relative to the project root.
SKIP_RELATIVE_PREFIXES = {
    Path("data/raw"),
    Path("data/processed"),
    Path("outputs/predictions"),
    Path("outputs/models"),
}

TEXT_EXTENSIONS = {
    ".json",
    ".csv",
    ".txt",
    ".yaml",
    ".yml",
    ".log",
    ".md",
    ".toml",
}

CODE_EXTENSIONS = {".py"}

HYPERPARAMETER_TERMS = (
    "best_params",
    "best_params_",
    "selected_params",
    "hyperparameters",
    "param_grid",
    "param_distributions",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_features",
    "class_weight",
    "learning_rate",
    "subsample",
    "l1_ratio",
    "random_state",
    "seed",
)

CODE_PATTERN = re.compile(
    r"GridSearchCV|RandomizedSearchCV|ParameterGrid|"
    r"best_params_?|selected_params|param_grid|param_distributions|"
    r"class_weight|random_state|\bseed\b|"
    r"n_estimators|max_depth|min_samples_leaf|min_samples_split|"
    r"max_features|learning_rate|subsample|l1_ratio|"
    r"(?<![A-Za-z0-9_])C(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

TARGET_SPLITS = {
    "random_firm_year",
    "random-firm-year",
    "random firm-year",
    "firm_grouped",
    "firm-grouped",
    "firm grouped",
    "time",
    "firm_grouped_time",
    "firm-grouped-time",
    "pre_covid_time",
    "pre-covid",
    "post2018_time",
    "post-2018",
}

PACKAGE_NAMES = (
    "scikit-learn",
    "shap",
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "joblib",
)


@dataclass
class Report:
    lines: list[str]

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def extend(self, values: Iterable[str]) -> None:
        self.lines.extend(values)

    def render(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit package versions, hyperparameter settings, random seeds, "
            "and model metrics without opening confidential microdata."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Replication-package root. By default, the directory containing "
            "this script is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diagnostics/hyperparameter_audit.txt"),
        help=(
            "Report path relative to --root, unless an absolute path is supplied. "
            "Default: outputs/diagnostics/hyperparameter_audit.txt"
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report without saving it to disk.",
    )
    parser.add_argument(
        "--max-text-kb",
        type=int,
        default=2048,
        help="Maximum size of each text file to inspect. Default: 2048 KB.",
    )
    parser.add_argument(
        "--max-code-kb",
        type=int,
        default=5120,
        help="Maximum size of each Python file to inspect. Default: 5120 KB.",
    )
    return parser.parse_args()


def normalise_root(root_arg: Path | None) -> Path:
    root = root_arg if root_arg is not None else Path(__file__).resolve().parent
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    return root


def is_skipped(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True

    if any(part in SKIP_DIR_NAMES for part in relative.parts):
        return True

    for prefix in SKIP_RELATIVE_PREFIXES:
        if relative == prefix or prefix in relative.parents:
            return True

    return False


def iter_files(
    root: Path,
    extensions: set[str],
    max_bytes: int,
) -> Iterator[Path]:
    for directory, dirs, files in os.walk(root):
        directory_path = Path(directory)

        dirs[:] = [
            name
            for name in dirs
            if name not in SKIP_DIR_NAMES
            and not is_skipped(root, directory_path / name)
        ]

        for filename in files:
            path = directory_path / filename
            if path.suffix.lower() not in extensions:
                continue
            if is_skipped(root, path):
                continue
            try:
                if path.stat().st_size <= max_bytes:
                    yield path
            except OSError:
                continue


def relative_display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed in this environment"
    except Exception as exc:  # pragma: no cover - defensive
        return f"unavailable ({type(exc).__name__})"


def add_versions(report: Report) -> None:
    report.add("=== ENVIRONMENT VERSIONS ===")
    report.add(f"python: {platform.python_version()}")
    report.add(f"implementation: {platform.python_implementation()}")
    report.add(f"platform: {platform.platform()}")
    for name in PACKAGE_NAMES:
        report.add(f"{name}: {package_version(name)}")
    report.add()


def relevant_lines(text: str) -> list[tuple[int, str]]:
    output: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if CODE_PATTERN.search(line):
            output.append((line_number, line.strip()))
    return output


def add_recorded_hyperparameters(
    report: Report,
    root: Path,
    max_bytes: int,
) -> int:
    report.add("=== TEXT OUTPUTS WITH RECORDED HYPERPARAMETERS ===")
    files_found = 0

    for path in iter_files(root, TEXT_EXTENSIONS, max_bytes):
        if path.name == "hyperparameter_audit.txt":
            continue

        try:
            text = read_text(path)
        except OSError:
            continue

        lower_text = text.lower()
        hits = sorted(
            {
                term
                for term in HYPERPARAMETER_TERMS
                if term.lower() in lower_text
            }
        )

        # Requiring at least two terms keeps the output focused.
        if len(hits) < 2:
            continue

        files_found += 1
        report.add()
        report.add(
            f"--- {relative_display(root, path)} "
            f"[matched: {', '.join(hits)}]"
        )

        if path.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
                pretty = json.dumps(obj, indent=2, ensure_ascii=False)
                report.add(pretty[:8000])
                if len(pretty) > 8000:
                    report.add("[JSON output truncated after 8,000 characters]")
                continue
            except json.JSONDecodeError:
                pass

        matches = relevant_lines(text)
        for line_number, line in matches[:120]:
            report.add(f"{line_number}: {line[:300]}")
        if len(matches) > 120:
            report.add(
                f"[{len(matches) - 120} additional matching lines omitted]"
            )

    if files_found == 0:
        report.add(
            "No small text output containing at least two hyperparameter "
            "keywords was found."
        )
    report.add()
    return files_found


def add_code_settings(
    report: Report,
    root: Path,
    max_bytes: int,
) -> int:
    report.add("=== CODE: SEARCH GRIDS, CLASS WEIGHTS, AND SEEDS ===")
    matches_found = 0

    for path in iter_files(root, CODE_EXTENSIONS, max_bytes):
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = read_text(path)
        except OSError:
            continue

        matches = relevant_lines(text)
        if not matches:
            continue

        report.add()
        report.add(f"--- {relative_display(root, path)}")
        for line_number, line in matches:
            matches_found += 1
            report.add(f"{line_number}: {line[:300]}")

    if matches_found == 0:
        report.add("No matching hyperparameter or seed settings were found.")
    report.add()
    return matches_found


def find_column(
    columns: Sequence[str],
    exact_candidates: Sequence[str],
    contains_candidates: Sequence[str] = (),
) -> str | None:
    lower_map = {column.lower(): column for column in columns}

    for candidate in exact_candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for column in columns:
        lower_column = column.lower()
        if any(fragment.lower() in lower_column for fragment in contains_candidates):
            return column

    return None


def metric_columns(columns: Sequence[str]) -> list[str]:
    fragments = (
        "roc_auc",
        "roc-auc",
        "auc",
        "pr_auc",
        "pr-auc",
        "lift",
        "precision",
        "p_at",
        "p@",
        "entry_rate",
        "baseline",
        "events",
        "positives",
        "test_n",
        "n_test",
    )
    selected: list[str] = []
    for column in columns:
        lower_column = column.lower()
        if any(fragment in lower_column for fragment in fragments):
            selected.append(column)
    return selected


def normalise_split(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", value.strip().lower())


def add_metrics_by_split(
    report: Report,
    root: Path,
    max_bytes: int,
) -> tuple[int, set[str]]:
    report.add("=== MODEL METRICS BY VALIDATION SPLIT ===")
    qualifying_files = 0
    all_splits: set[str] = set()

    for path in iter_files(root, {".csv"}, max_bytes):
        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="ignore",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                if not columns:
                    continue

                split_column = find_column(
                    columns,
                    exact_candidates=(
                        "split_design",
                        "split",
                        "split_name",
                        "validation_split",
                        "validation_design",
                    ),
                    contains_candidates=("split",),
                )
                metrics = metric_columns(columns)

                if split_column is None or not metrics:
                    continue

                rows = list(reader)
        except (OSError, csv.Error):
            continue

        split_values = sorted(
            {
                str(row.get(split_column, "")).strip()
                for row in rows
                if str(row.get(split_column, "")).strip()
            }
        )
        if not split_values:
            continue

        qualifying_files += 1
        normalised_values = {normalise_split(value) for value in split_values}
        all_splits.update(normalised_values)

        report.add()
        report.add(f"--- {relative_display(root, path)}")
        report.add(f"split column: {split_column}")
        report.add(f"splits found: {', '.join(split_values)}")
        report.add(f"metric columns: {', '.join(metrics)}")

        model_column = find_column(
            columns,
            exact_candidates=("model", "model_name", "estimator"),
            contains_candidates=("model", "estimator"),
        )

        target_rows = [
            row
            for row in rows
            if normalise_split(str(row.get(split_column, ""))) in TARGET_SPLITS
        ]

        if target_rows:
            report.add("candidate rows:")
            display_columns = [split_column]
            if model_column and model_column not in display_columns:
                display_columns.append(model_column)
            display_columns.extend(
                column for column in metrics if column not in display_columns
            )

            for row in target_rows[:80]:
                values = [
                    f"{column}={str(row.get(column, '')).strip()}"
                    for column in display_columns
                    if str(row.get(column, "")).strip()
                ]
                report.add("  " + " | ".join(values))
            if len(target_rows) > 80:
                report.add(
                    f"  [{len(target_rows) - 80} additional rows omitted]"
                )

    if qualifying_files == 0:
        report.add(
            "No small CSV file with both a split column and model-performance "
            "metrics was found."
        )
    report.add()
    return qualifying_files, all_splits


def add_split_summary(report: Report, all_splits: set[str]) -> None:
    report.add("=== SPLIT COVERAGE SUMMARY ===")
    requested = {
        "random_firm_year": (
            "random_firm_year",
            "random_firm_year_split",
        ),
        "firm_grouped": (
            "firm_grouped",
            "firm_grouped_split",
        ),
        "time": ("time", "chronological", "main_time"),
        "firm_grouped_time": (
            "firm_grouped_time",
            "strict",
            "strict_split",
        ),
    }

    for label, aliases in requested.items():
        present = any(alias in all_splits for alias in aliases)
        report.add(f"{label}: {'FOUND' if present else 'NOT FOUND'}")

    report.add(
        "Interpret NOT FOUND as 'not located in the inspected small CSV files', "
        "not as proof that the split was never estimated."
    )
    report.add()


def add_safety_summary(report: Report) -> None:
    report.add("=== SAFETY SCOPE ===")
    report.add("The audit intentionally skipped:")
    for prefix in sorted(str(item) for item in SKIP_RELATIVE_PREFIXES):
        report.add(f"- {prefix}")
    report.add(
        "It did not open Parquet, RData, DTA, pickle, joblib, or fitted-model files."
    )
    report.add()


def resolve_output(root: Path, output_arg: Path) -> Path:
    output = output_arg.expanduser()
    if not output.is_absolute():
        output = root / output
    return output.resolve()


def main() -> int:
    args = parse_args()

    try:
        root = normalise_root(args.root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = Report(lines=[])
    report.add("REPLICATION-PACKAGE HYPERPARAMETER AUDIT")
    report.add(f"project root folder: {root.name}")
    report.add()

    add_versions(report)
    recorded_count = add_recorded_hyperparameters(
        report,
        root,
        args.max_text_kb * 1024,
    )
    code_match_count = add_code_settings(
        report,
        root,
        args.max_code_kb * 1024,
    )
    metric_file_count, splits = add_metrics_by_split(
        report,
        root,
        args.max_text_kb * 1024,
    )
    add_split_summary(report, splits)
    add_safety_summary(report)

    report.add("=== AUDIT COUNTS ===")
    report.add(f"text outputs with hyperparameter evidence: {recorded_count}")
    report.add(f"matching code lines: {code_match_count}")
    report.add(f"metric CSV files with split information: {metric_file_count}")
    report.add()

    text = report.render()
    print(text, end="")

    if not args.no_write:
        output_path = resolve_output(root, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved audit report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
