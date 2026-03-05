#!/usr/bin/env python3
"""SciRS2 Performance Regression Detection Script.

Reads criterion benchmark JSON outputs from target/criterion/*/new/estimates.json,
compares mean.point_estimate values against stored baselines, and reports regressions.

Usage:
    python benches/regression_check.py
    python benches/regression_check.py --update-baseline
    python benches/regression_check.py --test
    python benches/regression_check.py --json-report --report-output /tmp/perf_report.json
    python benches/regression_check.py --pr-comment
"""

import json
import os
import sys
import unittest
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Regression alert threshold: >10% slower => REGRESSION
REGRESSION_THRESHOLD = 0.10
# Improvement alert threshold: >2% faster => IMPROVED
IMPROVEMENT_THRESHOLD = 0.02

STATUS_PASS = "PASS"
STATUS_REGRESSION = "REGRESSION"
STATUS_IMPROVED = "IMPROVED"
STATUS_NEW = "NEW"


def format_ns(ns: float) -> str:
    """Format a nanosecond value into a human-readable string."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f}  s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.1f} ns"


def discover_criterion_results(criterion_dir: Path) -> dict:
    """Walk criterion output directory and collect mean point estimates (ns).

    Criterion layout: <criterion_dir>/<group>/<bench>/new/estimates.json
    Returns dict mapping test_name -> mean_ns (float).
    """
    results = {}
    if not criterion_dir.is_dir():
        return results
    for estimates_path in criterion_dir.rglob("new/estimates.json"):
        try:
            with estimates_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not read {estimates_path}: {exc}", file=sys.stderr)
            continue
        mean_block = data.get("mean")
        if not isinstance(mean_block, dict) or "point_estimate" not in mean_block:
            print(f"Warning: no mean.point_estimate in {estimates_path}", file=sys.stderr)
            continue
        try:
            rel = estimates_path.relative_to(criterion_dir)
        except ValueError:
            rel = estimates_path
        # Drop trailing "new" and "estimates.json" to form the test name
        test_name = "/".join(list(rel.parts)[:-2])
        results[test_name] = float(mean_block["point_estimate"])
    return results


def load_baselines(path: Path) -> dict:
    """Load baseline JSON from disk; return empty dict on any error."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load baselines from {path}: {exc}", file=sys.stderr)
        return {}


def save_baselines(path: Path, current: dict, existing: dict) -> None:
    """Merge current results into existing baselines and persist to disk."""
    now = datetime.now(timezone.utc).isoformat()
    merged = {**existing, **{k: {"mean_ns": v, "timestamp": now} for k, v in current.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        print(f"Baselines saved to {path} ({len(merged)} entries).")
    except OSError as exc:
        print(f"Error: could not save baselines: {exc}", file=sys.stderr)


def compare_results(current: dict, baselines: dict) -> list:
    """Return a list of comparison rows (dicts) for every current benchmark."""
    rows = []
    for test_name in sorted(current):
        current_ns = current[test_name]
        entry = baselines.get(test_name)
        baseline_ns = entry.get("mean_ns") if isinstance(entry, dict) else None
        if baseline_ns is None or baseline_ns <= 0:
            rows.append({
                "status": STATUS_NEW,
                "test_name": test_name,
                "current_ns": current_ns,
                "baseline_ns": None,
                "pct_change": None,
            })
            continue
        pct = (current_ns - baseline_ns) / baseline_ns
        if pct > REGRESSION_THRESHOLD:
            status = STATUS_REGRESSION
        elif pct < -IMPROVEMENT_THRESHOLD:
            status = STATUS_IMPROVED
        else:
            status = STATUS_PASS
        rows.append({
            "status": status,
            "test_name": test_name,
            "current_ns": current_ns,
            "baseline_ns": baseline_ns,
            "pct_change": pct,
        })
    return rows


def build_json_report(rows: list, date_str: str) -> dict:
    """Build a structured JSON report from comparison rows.

    Each entry in 'benchmarks' has:
        timestamp          - ISO-8601 UTC timestamp of this run
        benchmark_name     - Criterion test path
        baseline_value     - Baseline mean time in nanoseconds (null if new)
        current_value      - Current mean time in nanoseconds
        change_percent     - Signed percentage change (null if new)
        status             - PASS | REGRESSION | IMPROVED | NEW
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    counts = {STATUS_PASS: 0, STATUS_REGRESSION: 0, STATUS_IMPROVED: 0, STATUS_NEW: 0}
    benchmarks = []
    for row in rows:
        counts[row["status"]] += 1
        change_pct = None
        if row["pct_change"] is not None:
            change_pct = round(row["pct_change"] * 100, 4)
        benchmarks.append({
            "timestamp": timestamp,
            "benchmark_name": row["test_name"],
            "baseline_value": row["baseline_ns"],
            "current_value": row["current_ns"],
            "change_percent": change_pct,
            "status": row["status"],
        })
    has_regressions = counts[STATUS_REGRESSION] > 0
    return {
        "report_date": date_str,
        "generated_at": timestamp,
        "regression_threshold_pct": REGRESSION_THRESHOLD * 100,
        "improvement_threshold_pct": IMPROVEMENT_THRESHOLD * 100,
        "summary": {
            "total": len(rows),
            "passed": counts[STATUS_PASS],
            "regressions": counts[STATUS_REGRESSION],
            "improved": counts[STATUS_IMPROVED],
            "new": counts[STATUS_NEW],
            "has_regressions": has_regressions,
        },
        "benchmarks": benchmarks,
    }


def save_json_report(report: dict, path: Path) -> None:
    """Write the JSON report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"JSON report saved to {path}.")
    except OSError as exc:
        print(f"Error: could not save JSON report: {exc}", file=sys.stderr)


def print_report(rows: list, date_str: str) -> None:
    """Print a formatted regression report table and summary to stdout."""
    title = f"SciRS2 Regression Check - {date_str}"
    print(title)
    print("=" * max(len(title), 40))
    counts = {STATUS_PASS: 0, STATUS_REGRESSION: 0, STATUS_IMPROVED: 0, STATUS_NEW: 0}
    col = max((len(r["test_name"]) for r in rows), default=30)
    for row in rows:
        status, name = row["status"], row["test_name"]
        counts[status] += 1
        cur = format_ns(row["current_ns"])
        if row["baseline_ns"] is not None:
            base = format_ns(row["baseline_ns"])
            sign = "+" if row["pct_change"] >= 0 else ""
            pct_str = f"{sign}{row['pct_change'] * 100:.1f}%"
            marker = " <- REGRESSION" if status == STATUS_REGRESSION else ""
            print(f"{status:<10} | {name:<{col}} | {cur:>10} | baseline {base:>10} | {pct_str:>7}{marker}")
        else:
            print(f"{status:<10} | {name:<{col}} | {cur:>10} | (no baseline)")
    print()
    total = sum(counts.values())
    print(f"Summary: {counts[STATUS_PASS]} passed, {counts[STATUS_REGRESSION]} regression(s), "
          f"{counts[STATUS_IMPROVED]} improved, {counts[STATUS_NEW]} new  (total: {total})")
    has_regressions = counts[STATUS_REGRESSION] > 0
    code = 1 if has_regressions else 0
    print(f"EXIT: {code} ({'regressions found' if has_regressions else 'all clear'})")


def print_pr_comment(report: dict) -> None:
    """Print a GitHub PR comment-formatted markdown summary to stdout."""
    summary = report["summary"]
    benchmarks = report["benchmarks"]
    date_str = report["report_date"]
    threshold_pct = report["regression_threshold_pct"]

    status_icon = ":x:" if summary["has_regressions"] else ":white_check_mark:"
    headline = "Performance regressions detected" if summary["has_regressions"] else "No performance regressions"

    lines = [
        f"## {status_icon} SciRS2 Performance Benchmark Report ({date_str})",
        "",
        f"**{headline}** (threshold: >{threshold_pct:.0f}% slower)",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total benchmarks | {summary['total']} |",
        f"| Passed | {summary['passed']} |",
        f"| Regressions | {summary['regressions']} |",
        f"| Improved | {summary['improved']} |",
        f"| New | {summary['new']} |",
        "",
    ]

    regressions = [b for b in benchmarks if b["status"] == STATUS_REGRESSION]
    if regressions:
        lines += [
            "### Regressions",
            "",
            "| Benchmark | Baseline | Current | Change |",
            "|-----------|----------|---------|--------|",
        ]
        for b in regressions:
            base_str = format_ns(b["baseline_value"]) if b["baseline_value"] is not None else "N/A"
            curr_str = format_ns(b["current_value"])
            chg_str = f"+{b['change_percent']:.1f}%" if b["change_percent"] is not None else "N/A"
            lines.append(f"| `{b['benchmark_name']}` | {base_str} | {curr_str} | {chg_str} |")
        lines.append("")

    improvements = [b for b in benchmarks if b["status"] == STATUS_IMPROVED]
    if improvements:
        lines += [
            "<details><summary>Improvements (" + str(len(improvements)) + ")</summary>",
            "",
            "| Benchmark | Baseline | Current | Change |",
            "|-----------|----------|---------|--------|",
        ]
        for b in improvements:
            base_str = format_ns(b["baseline_value"]) if b["baseline_value"] is not None else "N/A"
            curr_str = format_ns(b["current_value"])
            chg_str = f"{b['change_percent']:.1f}%" if b["change_percent"] is not None else "N/A"
            lines.append(f"| `{b['benchmark_name']}` | {base_str} | {curr_str} | {chg_str} |")
        lines += ["", "</details>", ""]

    lines += [
        "---",
        "_Generated by SciRS2 regression_check.py_",
    ]

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class RegressionCheckTests(unittest.TestCase):
    def _baseline(self, ns):
        return {"mean_ns": ns, "timestamp": "2026-01-01T00:00:00+00:00"}

    def test_format_ns_units(self):
        self.assertIn("ns", format_ns(85.3))
        self.assertIn("us", format_ns(1_500))
        self.assertIn("ms", format_ns(1_500_000))
        self.assertIn("s", format_ns(2_000_000_000))

    def test_pass(self):
        rows = compare_results({"b": 100.0}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_PASS)

    def test_regression_10pct(self):
        # Exactly at new 10% threshold — strictly greater, so PASS
        rows = compare_results({"b": 110.0}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_PASS)

    def test_regression_over_10pct(self):
        # 10.1% increase => REGRESSION
        rows = compare_results({"b": 110.1}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_REGRESSION)
        self.assertGreater(rows[0]["pct_change"], 0.10)

    def test_regression_pct_change_value(self):
        rows = compare_results({"b": 120.0}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_REGRESSION)
        self.assertAlmostEqual(rows[0]["pct_change"], 0.2, places=5)

    def test_improved(self):
        rows = compare_results({"b": 70.0}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_IMPROVED)

    def test_new(self):
        rows = compare_results({"b": 50.0}, {})
        self.assertEqual(rows[0]["status"], STATUS_NEW)
        self.assertIsNone(rows[0]["baseline_ns"])

    def test_borderline_not_regression(self):
        # Exactly 10% — threshold is strictly greater than, so PASS
        rows = compare_results({"b": 110.0}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_PASS)

    def test_just_over_threshold_is_regression(self):
        rows = compare_results({"b": 110.1}, {"b": self._baseline(100.0)})
        self.assertEqual(rows[0]["status"], STATUS_REGRESSION)

    def test_discover_valid_estimates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir) / "simd_f64_sum" / "new"
            bench_dir.mkdir(parents=True)
            (bench_dir / "estimates.json").write_text(
                json.dumps({"mean": {"point_estimate": 85300.0}}))
            results = discover_criterion_results(Path(tmpdir))
            self.assertIn("simd_f64_sum", results)
            self.assertAlmostEqual(results["simd_f64_sum"], 85300.0)

    def test_discover_missing_directory(self):
        self.assertEqual(discover_criterion_results(Path("/nonexistent")), {})

    def test_discover_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "bad" / "new"
            d.mkdir(parents=True)
            (d / "estimates.json").write_text("not json")
            self.assertEqual(discover_criterion_results(Path(tmpdir)), {})

    def test_baselines_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baselines.json"
            save_baselines(path, {"x": 123.4}, {})
            loaded = load_baselines(path)
            self.assertAlmostEqual(loaded["x"]["mean_ns"], 123.4)

    def test_baselines_missing_file(self):
        self.assertEqual(load_baselines(Path("/nonexistent/baselines.json")), {})

    def test_baselines_merge_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baselines.json"
            existing = {"old": {"mean_ns": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"}}
            save_baselines(path, {"new_bench": 75.0}, existing)
            loaded = load_baselines(path)
            self.assertIn("old", loaded)
            self.assertIn("new_bench", loaded)

    def test_json_report_structure(self):
        rows = [
            {"status": STATUS_PASS, "test_name": "a/bench", "current_ns": 100.0,
             "baseline_ns": 95.0, "pct_change": 0.0526},
            {"status": STATUS_REGRESSION, "test_name": "b/bench", "current_ns": 200.0,
             "baseline_ns": 150.0, "pct_change": 0.333},
            {"status": STATUS_NEW, "test_name": "c/bench", "current_ns": 50.0,
             "baseline_ns": None, "pct_change": None},
        ]
        report = build_json_report(rows, "2026-02-21")
        self.assertEqual(report["summary"]["total"], 3)
        self.assertEqual(report["summary"]["regressions"], 1)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["new"], 1)
        self.assertTrue(report["summary"]["has_regressions"])
        benchmarks = {b["benchmark_name"]: b for b in report["benchmarks"]}
        self.assertIn("a/bench", benchmarks)
        self.assertIn("b/bench", benchmarks)
        self.assertIn("c/bench", benchmarks)
        self.assertIsNone(benchmarks["c/bench"]["baseline_value"])
        self.assertIsNone(benchmarks["c/bench"]["change_percent"])
        self.assertEqual(benchmarks["b/bench"]["status"], STATUS_REGRESSION)
        # change_percent is rounded to 4 decimal places
        self.assertAlmostEqual(benchmarks["a/bench"]["change_percent"], 5.26, places=1)

    def test_json_report_keys(self):
        """Every benchmark entry must have exactly the required keys."""
        rows = [
            {"status": STATUS_PASS, "test_name": "x", "current_ns": 1.0,
             "baseline_ns": 1.0, "pct_change": 0.0},
        ]
        report = build_json_report(rows, "2026-02-21")
        entry = report["benchmarks"][0]
        required_keys = {"timestamp", "benchmark_name", "baseline_value",
                         "current_value", "change_percent", "status"}
        self.assertEqual(set(entry.keys()), required_keys)

    def test_json_report_no_regressions(self):
        rows = [
            {"status": STATUS_PASS, "test_name": "x", "current_ns": 100.0,
             "baseline_ns": 100.0, "pct_change": 0.0},
        ]
        report = build_json_report(rows, "2026-02-21")
        self.assertFalse(report["summary"]["has_regressions"])

    def test_json_report_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            rows = [{"status": STATUS_IMPROVED, "test_name": "fast/op",
                     "current_ns": 80.0, "baseline_ns": 100.0, "pct_change": -0.2}]
            report = build_json_report(rows, "2026-02-21")
            save_json_report(report, path)
            with path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["improved"], 1)
            self.assertEqual(loaded["benchmarks"][0]["benchmark_name"], "fast/op")

    def test_regression_threshold_is_10pct(self):
        """Ensure the global threshold constant matches the 10% requirement."""
        self.assertAlmostEqual(REGRESSION_THRESHOLD, 0.10, places=5)

    def test_pr_comment_contains_regression_table(self):
        """PR comment output should include regression table when regressions exist."""
        import io
        rows = [
            {"status": STATUS_REGRESSION, "test_name": "slow/op", "current_ns": 220.0,
             "baseline_ns": 100.0, "pct_change": 1.2},
        ]
        report = build_json_report(rows, "2026-02-21")
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            print_pr_comment(report)
        finally:
            sys.stdout = old_stdout
        output = buffer.getvalue()
        self.assertIn("Regressions", output)
        self.assertIn("slow/op", output)
        self.assertIn(":x:", output)

    def test_pr_comment_no_regression_checkmark(self):
        """PR comment shows checkmark when no regressions."""
        import io
        rows = [
            {"status": STATUS_PASS, "test_name": "ok/op", "current_ns": 100.0,
             "baseline_ns": 100.0, "pct_change": 0.0},
        ]
        report = build_json_report(rows, "2026-02-21")
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            print_pr_comment(report)
        finally:
            sys.stdout = old_stdout
        output = buffer.getvalue()
        self.assertIn(":white_check_mark:", output)
        self.assertNotIn(":x:", output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="SciRS2 benchmark regression detector.")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Save current results as new baseline; skip regression check.")
    parser.add_argument("--test", action="store_true",
                        help="Run built-in unit tests.")
    parser.add_argument("--baselines", default=None,
                        help="Path to baselines JSON (default: benches/performance_baseline.json).")
    parser.add_argument("--criterion-dir", default=None,
                        help="Path to criterion output dir (default: target/criterion).")
    parser.add_argument("--json-report", action="store_true",
                        help="Write a structured JSON report to disk.")
    parser.add_argument("--report-output", default=None,
                        help="Path for the JSON report output. "
                             "Defaults to /tmp/scirs2_perf_report_YYYYMMDD.json. "
                             "Only used when --json-report is given.")
    parser.add_argument("--pr-comment", action="store_true",
                        help="Print a GitHub PR comment-formatted markdown summary to stdout.")
    args = parser.parse_args()

    if args.test:
        sys.argv = [sys.argv[0]]
        suite = unittest.TestLoader().loadTestsFromTestCase(RegressionCheckTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    baselines_path = (
        Path(args.baselines) if args.baselines
        else script_dir / "performance_baseline.json"
    )
    criterion_dir = (
        Path(args.criterion_dir) if args.criterion_dir
        else repo_root / "target" / "criterion"
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Criterion dir : {criterion_dir}")
    print(f"Baselines file: {baselines_path}")
    print()

    current = discover_criterion_results(criterion_dir)
    if not current:
        print("No criterion results found.  Run `cargo bench -p scirs2-benchmarks` first.",
              file=sys.stderr)
        return 2

    baselines = load_baselines(baselines_path)

    if args.update_baseline:
        save_baselines(baselines_path, current, baselines)
        print("Baseline updated.  Exiting without regression check.")
        return 0

    rows = compare_results(current, baselines)
    if not rows:
        print("No results to compare.")
        return 0

    print_report(rows, date_str)

    report = build_json_report(rows, date_str)

    if args.json_report:
        if args.report_output:
            report_path = Path(args.report_output)
        else:
            report_path = Path(f"/tmp/scirs2_perf_report_{date_str}.json")
        save_json_report(report, report_path)

    if args.pr_comment:
        print_pr_comment(report)

    return 1 if report["summary"]["has_regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
