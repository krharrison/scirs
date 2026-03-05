# SciRS2 Benchmark Suite (v0.3.0)

Performance benchmarks using [Criterion](https://github.com/bheisler/criterion.rs).

## Running Benchmarks

```bash
cargo bench -p scirs2-benchmarks
cargo bench -p scirs2-benchmarks -- simd_f64   # filter by name
```

Results land in `target/criterion/`.

## Regression Detection

```bash
# Check against stored baseline
python benches/regression_check.py

# Save current results as new baseline
python benches/regression_check.py --update-baseline

# Run built-in unit tests
python benches/regression_check.py --test

# Custom paths
python benches/regression_check.py \
    --criterion-dir target/criterion \
    --baselines benches/baselines.json
```

## Output Format

```
SciRS2 Regression Check - 2026-02-17
=====================================
PASS       | simd_f64_sum_1000  |   85.3 ns | baseline   83.1 ns |   +2.6%
IMPROVED   | simd_f32_dot_512   |  120.1 ns | baseline  150.2 ns |  -20.1%
REGRESSION | matrix_mul_1000x1000| 500.2 ms | baseline  450.1 ms |  +11.1% <- REGRESSION

Summary: 47 passed, 1 regression(s), 3 improved, 5 new  (total: 56)
EXIT: 1 (regressions found)
```

Exit code `1` means regressions detected; `0` means all clear; `2` means no data found.

## Baseline Storage

Baselines live in `benches/baselines.json`:

```json
{
  "simd_f64_sum_1000": { "mean_ns": 85300.0, "timestamp": "2026-02-17T10:00:00+00:00" }
}
```

Commit this file to preserve performance history across branches.

## Dependencies

The Python script uses only stdlib: `json`, `os`, `sys`, `pathlib`, `argparse`,
`datetime`, `unittest`.  No third-party packages required.
