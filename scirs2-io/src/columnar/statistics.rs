//! Column-level statistics and row group support for the columnar format.
//!
//! Provides:
//! - Per-column statistics (min, max, count, null_count, sum)
//! - Row group splitting for large tables
//! - Predicate pushdown support (column filtering during read)

use crate::error::{IoError, Result};

use super::types::{Column, ColumnData, ColumnarTable};

// =============================================================================
// Column statistics
// =============================================================================

/// Statistics for a single column
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Number of values
    pub count: usize,
    /// Number of null/missing values (always 0 for now, extensible)
    pub null_count: usize,
    /// Minimum value (as f64, NaN for non-numeric)
    pub min: Option<f64>,
    /// Maximum value (as f64, NaN for non-numeric)
    pub max: Option<f64>,
    /// Sum of values (for numeric columns)
    pub sum: Option<f64>,
    /// Number of distinct values
    pub distinct_count: Option<usize>,
}

impl ColumnStats {
    /// Compute statistics for a column
    pub fn from_column(col: &Column) -> Self {
        let count = col.len();
        let null_count = 0; // extensible for future Option<T> columns

        match &col.data {
            ColumnData::Float64(v) => {
                let (min, max, sum) = if v.is_empty() {
                    (None, None, None)
                } else {
                    let mut mn = f64::INFINITY;
                    let mut mx = f64::NEG_INFINITY;
                    let mut s = 0.0;
                    for &val in v {
                        if val < mn {
                            mn = val;
                        }
                        if val > mx {
                            mx = val;
                        }
                        s += val;
                    }
                    (Some(mn), Some(mx), Some(s))
                };
                ColumnStats {
                    name: col.name.clone(),
                    count,
                    null_count,
                    min,
                    max,
                    sum,
                    distinct_count: None,
                }
            }
            ColumnData::Int64(v) => {
                let (min, max, sum) = if v.is_empty() {
                    (None, None, None)
                } else {
                    let mut mn = i64::MAX;
                    let mut mx = i64::MIN;
                    let mut s: i64 = 0;
                    for &val in v {
                        if val < mn {
                            mn = val;
                        }
                        if val > mx {
                            mx = val;
                        }
                        s = s.wrapping_add(val);
                    }
                    (Some(mn as f64), Some(mx as f64), Some(s as f64))
                };
                ColumnStats {
                    name: col.name.clone(),
                    count,
                    null_count,
                    min,
                    max,
                    sum,
                    distinct_count: None,
                }
            }
            ColumnData::Str(v) => {
                let distinct = {
                    let mut set = std::collections::HashSet::new();
                    for s in v {
                        set.insert(s.as_str());
                    }
                    set.len()
                };
                ColumnStats {
                    name: col.name.clone(),
                    count,
                    null_count,
                    min: None,
                    max: None,
                    sum: None,
                    distinct_count: Some(distinct),
                }
            }
            ColumnData::Bool(v) => {
                let true_count = v.iter().filter(|&&b| b).count();
                ColumnStats {
                    name: col.name.clone(),
                    count,
                    null_count,
                    min: Some(0.0),
                    max: Some(1.0),
                    sum: Some(true_count as f64),
                    distinct_count: Some(if v.is_empty() {
                        0
                    } else if true_count == 0 || true_count == count {
                        1
                    } else {
                        2
                    }),
                }
            }
        }
    }

    /// Check if a numeric value could exist in this column based on min/max
    pub fn could_contain_value(&self, value: f64) -> bool {
        match (self.min, self.max) {
            (Some(mn), Some(mx)) => value >= mn && value <= mx,
            _ => true, // Non-numeric or empty -- can't rule out
        }
    }
}

/// Statistics for all columns in a table
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Per-column statistics
    pub columns: Vec<ColumnStats>,
    /// Total number of rows
    pub num_rows: usize,
}

impl TableStats {
    /// Compute statistics for all columns in a table
    pub fn from_table(table: &ColumnarTable) -> Self {
        let columns = table
            .columns()
            .iter()
            .map(|col| ColumnStats::from_column(col))
            .collect();
        TableStats {
            columns,
            num_rows: table.num_rows(),
        }
    }

    /// Get stats for a column by name
    pub fn column_stats(&self, name: &str) -> Option<&ColumnStats> {
        self.columns.iter().find(|cs| cs.name == name)
    }
}

// =============================================================================
// Row groups
// =============================================================================

/// A row group is a horizontal partition of the table (a range of rows).
/// Each row group stores the same columns but over a subset of rows.
#[derive(Debug, Clone)]
pub struct RowGroup {
    /// Starting row index (inclusive)
    pub start_row: usize,
    /// Number of rows in this group
    pub num_rows: usize,
    /// Per-column statistics for this row group
    pub stats: Vec<ColumnStats>,
}

/// Configuration for row-group splitting
#[derive(Debug, Clone)]
pub struct RowGroupConfig {
    /// Maximum number of rows per row group
    pub max_rows_per_group: usize,
}

impl Default for RowGroupConfig {
    fn default() -> Self {
        RowGroupConfig {
            max_rows_per_group: 65_536,
        }
    }
}

/// Split a table into row groups with per-group statistics
pub fn split_into_row_groups(
    table: &ColumnarTable,
    config: &RowGroupConfig,
) -> Result<Vec<RowGroup>> {
    let total_rows = table.num_rows();
    if total_rows == 0 {
        return Ok(Vec::new());
    }

    let max_per = config.max_rows_per_group.max(1);
    let num_groups = (total_rows + max_per - 1) / max_per;
    let mut groups = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        let start = g * max_per;
        let end = (start + max_per).min(total_rows);
        let group_rows = end - start;

        // Compute per-column stats for this row group
        let stats: Vec<ColumnStats> = table
            .columns()
            .iter()
            .map(|col| {
                let slice_col = slice_column(col, start, end);
                ColumnStats::from_column(&slice_col)
            })
            .collect();

        groups.push(RowGroup {
            start_row: start,
            num_rows: group_rows,
            stats,
        });
    }

    Ok(groups)
}

/// Extract a row-range slice from a column
fn slice_column(col: &Column, start: usize, end: usize) -> Column {
    let data = match &col.data {
        ColumnData::Float64(v) => ColumnData::Float64(v[start..end].to_vec()),
        ColumnData::Int64(v) => ColumnData::Int64(v[start..end].to_vec()),
        ColumnData::Str(v) => ColumnData::Str(v[start..end].to_vec()),
        ColumnData::Bool(v) => ColumnData::Bool(v[start..end].to_vec()),
    };
    Column {
        name: col.name.clone(),
        data,
    }
}

/// Extract a sub-table for a specific row group
pub fn extract_row_group(table: &ColumnarTable, group: &RowGroup) -> Result<ColumnarTable> {
    let start = group.start_row;
    let end = start + group.num_rows;

    let columns: Vec<Column> = table
        .columns()
        .iter()
        .map(|col| slice_column(col, start, end))
        .collect();

    ColumnarTable::from_columns(columns)
}

// =============================================================================
// Predicate pushdown (column filtering)
// =============================================================================

/// A predicate for filtering rows or skipping row groups
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Column value equals a given f64
    FloatEquals(String, f64),
    /// Column value is in range [lo, hi]
    FloatRange(String, f64, f64),
    /// Column i64 value equals
    IntEquals(String, i64),
    /// Column i64 value is in range [lo, hi]
    IntRange(String, i64, i64),
    /// Column string value equals
    StrEquals(String, String),
    /// Column bool value equals
    BoolEquals(String, bool),
    /// Logical AND of predicates
    And(Vec<Predicate>),
    /// Logical OR of predicates
    Or(Vec<Predicate>),
}

impl Predicate {
    /// Check whether a row group *could* contain matching rows
    /// based on its column statistics (predicate pushdown).
    /// Returns true if the group cannot be ruled out.
    pub fn could_match_row_group(&self, group: &RowGroup) -> bool {
        match self {
            Predicate::FloatEquals(col_name, val) => {
                if let Some(stats) = group.stats.iter().find(|s| s.name == *col_name) {
                    stats.could_contain_value(*val)
                } else {
                    true // column not found, can't rule out
                }
            }
            Predicate::FloatRange(col_name, lo, hi) => {
                if let Some(stats) = group.stats.iter().find(|s| s.name == *col_name) {
                    match (stats.min, stats.max) {
                        (Some(mn), Some(mx)) => mx >= *lo && mn <= *hi,
                        _ => true,
                    }
                } else {
                    true
                }
            }
            Predicate::IntEquals(col_name, val) => {
                if let Some(stats) = group.stats.iter().find(|s| s.name == *col_name) {
                    stats.could_contain_value(*val as f64)
                } else {
                    true
                }
            }
            Predicate::IntRange(col_name, lo, hi) => {
                if let Some(stats) = group.stats.iter().find(|s| s.name == *col_name) {
                    match (stats.min, stats.max) {
                        (Some(mn), Some(mx)) => mx >= *lo as f64 && mn <= *hi as f64,
                        _ => true,
                    }
                } else {
                    true
                }
            }
            Predicate::StrEquals(_col_name, _val) => {
                // Can't prune string equality from min/max; always matches
                true
            }
            Predicate::BoolEquals(col_name, val) => {
                if let Some(stats) = group.stats.iter().find(|s| s.name == *col_name) {
                    if let Some(sum) = stats.sum {
                        if *val {
                            sum > 0.0 // at least one true
                        } else {
                            sum < stats.count as f64 // at least one false
                        }
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            Predicate::And(preds) => preds.iter().all(|p| p.could_match_row_group(group)),
            Predicate::Or(preds) => preds.iter().any(|p| p.could_match_row_group(group)),
        }
    }

    /// Evaluate this predicate row-by-row against a table,
    /// returning a boolean mask of matching rows.
    pub fn evaluate(&self, table: &ColumnarTable) -> Result<Vec<bool>> {
        let n = table.num_rows();
        match self {
            Predicate::FloatEquals(col_name, val) => {
                let data = table.get_f64(col_name)?;
                Ok(data
                    .iter()
                    .map(|&v| (v - val).abs() < f64::EPSILON)
                    .collect())
            }
            Predicate::FloatRange(col_name, lo, hi) => {
                let data = table.get_f64(col_name)?;
                Ok(data.iter().map(|&v| v >= *lo && v <= *hi).collect())
            }
            Predicate::IntEquals(col_name, val) => {
                let data = table.get_i64(col_name)?;
                Ok(data.iter().map(|&v| v == *val).collect())
            }
            Predicate::IntRange(col_name, lo, hi) => {
                let data = table.get_i64(col_name)?;
                Ok(data.iter().map(|&v| v >= *lo && v <= *hi).collect())
            }
            Predicate::StrEquals(col_name, val) => {
                let data = table.get_str(col_name)?;
                Ok(data.iter().map(|v| v == val).collect())
            }
            Predicate::BoolEquals(col_name, val) => {
                let data = table.get_bool(col_name)?;
                Ok(data.iter().map(|&v| v == *val).collect())
            }
            Predicate::And(preds) => {
                let mut result = vec![true; n];
                for p in preds {
                    let mask = p.evaluate(table)?;
                    for (r, m) in result.iter_mut().zip(mask.iter()) {
                        *r = *r && *m;
                    }
                }
                Ok(result)
            }
            Predicate::Or(preds) => {
                let mut result = vec![false; n];
                for p in preds {
                    let mask = p.evaluate(table)?;
                    for (r, m) in result.iter_mut().zip(mask.iter()) {
                        *r = *r || *m;
                    }
                }
                Ok(result)
            }
        }
    }
}

/// Read a columnar file with column selection (projection pushdown)
pub fn read_columnar_with_columns<P: AsRef<std::path::Path>>(
    path: P,
    columns: &[&str],
) -> Result<ColumnarTable> {
    let full = super::reader::read_columnar(path)?;
    select_columns(&full, columns)
}

/// Select a subset of columns from a table
pub fn select_columns(table: &ColumnarTable, columns: &[&str]) -> Result<ColumnarTable> {
    let mut selected = Vec::with_capacity(columns.len());
    for &name in columns {
        let col = table.column(name)?;
        selected.push(col.clone());
    }
    ColumnarTable::from_columns(selected)
}

/// Filter a table to only rows matching a predicate
pub fn filter_table(table: &ColumnarTable, predicate: &Predicate) -> Result<ColumnarTable> {
    let mask = predicate.evaluate(table)?;
    let mut columns = Vec::with_capacity(table.num_columns());

    for col in table.columns() {
        let filtered_data = match &col.data {
            ColumnData::Float64(v) => {
                let filtered: Vec<f64> = v
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect();
                ColumnData::Float64(filtered)
            }
            ColumnData::Int64(v) => {
                let filtered: Vec<i64> = v
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect();
                ColumnData::Int64(filtered)
            }
            ColumnData::Str(v) => {
                let filtered: Vec<String> = v
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(val, _)| val.clone())
                    .collect();
                ColumnData::Str(filtered)
            }
            ColumnData::Bool(v) => {
                let filtered: Vec<bool> = v
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect();
                ColumnData::Bool(filtered)
            }
        };
        columns.push(Column {
            name: col.name.clone(),
            data: filtered_data,
        });
    }

    ColumnarTable::from_columns(columns)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_table() -> ColumnarTable {
        ColumnarTable::from_columns(vec![
            Column::float64("temp", vec![20.0, 22.5, 18.0, 25.0, 19.5]),
            Column::int64("id", vec![1, 2, 3, 4, 5]),
            Column::string(
                "city",
                vec![
                    "Tokyo".into(),
                    "Osaka".into(),
                    "Tokyo".into(),
                    "Kyoto".into(),
                    "Osaka".into(),
                ],
            ),
            Column::boolean("active", vec![true, true, false, true, false]),
        ])
        .expect("table creation failed")
    }

    #[test]
    fn test_column_stats_float64() {
        let col = Column::float64("temp", vec![20.0, 22.5, 18.0, 25.0, 19.5]);
        let stats = ColumnStats::from_column(&col);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.null_count, 0);
        assert!((stats.min.expect("no min") - 18.0).abs() < 1e-10);
        assert!((stats.max.expect("no max") - 25.0).abs() < 1e-10);
        assert!((stats.sum.expect("no sum") - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_stats_int64() {
        let col = Column::int64("id", vec![1, 2, 3, 4, 5]);
        let stats = ColumnStats::from_column(&col);
        assert_eq!(stats.count, 5);
        assert!((stats.min.expect("no min") - 1.0).abs() < 1e-10);
        assert!((stats.max.expect("no max") - 5.0).abs() < 1e-10);
        assert!((stats.sum.expect("no sum") - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_stats_string() {
        let col = Column::string("city", vec!["a".into(), "b".into(), "a".into(), "c".into()]);
        let stats = ColumnStats::from_column(&col);
        assert_eq!(stats.count, 4);
        assert!(stats.min.is_none());
        assert!(stats.max.is_none());
        assert_eq!(stats.distinct_count, Some(3));
    }

    #[test]
    fn test_column_stats_bool() {
        let col = Column::boolean("flags", vec![true, false, true, true, false]);
        let stats = ColumnStats::from_column(&col);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.distinct_count, Some(2));
        assert!((stats.sum.expect("no sum") - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_table_stats() {
        let table = make_test_table();
        let stats = TableStats::from_table(&table);
        assert_eq!(stats.num_rows, 5);
        assert_eq!(stats.columns.len(), 4);

        let temp_stats = stats.column_stats("temp").expect("temp stats missing");
        assert!((temp_stats.min.expect("no min") - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_row_group_split() {
        let table = make_test_table();
        let config = RowGroupConfig {
            max_rows_per_group: 2,
        };
        let groups = split_into_row_groups(&table, &config).expect("split failed");

        // 5 rows / 2 per group => 3 groups (2, 2, 1)
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].start_row, 0);
        assert_eq!(groups[0].num_rows, 2);
        assert_eq!(groups[1].start_row, 2);
        assert_eq!(groups[1].num_rows, 2);
        assert_eq!(groups[2].start_row, 4);
        assert_eq!(groups[2].num_rows, 1);
    }

    #[test]
    fn test_row_group_stats() {
        let table = make_test_table();
        let config = RowGroupConfig {
            max_rows_per_group: 3,
        };
        let groups = split_into_row_groups(&table, &config).expect("split failed");

        // First group: rows 0..3, temp=[20.0, 22.5, 18.0]
        let g0_temp = groups[0]
            .stats
            .iter()
            .find(|s| s.name == "temp")
            .expect("temp stats");
        assert!((g0_temp.min.expect("no min") - 18.0).abs() < 1e-10);
        assert!((g0_temp.max.expect("no max") - 22.5).abs() < 1e-10);
    }

    #[test]
    fn test_extract_row_group() {
        let table = make_test_table();
        let config = RowGroupConfig {
            max_rows_per_group: 2,
        };
        let groups = split_into_row_groups(&table, &config).expect("split failed");

        let sub = extract_row_group(&table, &groups[1]).expect("extract failed");
        assert_eq!(sub.num_rows(), 2);
        let ids = sub.get_i64("id").expect("get id failed");
        assert_eq!(ids, &[3, 4]);
    }

    #[test]
    fn test_predicate_pushdown_float_range() {
        let table = make_test_table();
        let config = RowGroupConfig {
            max_rows_per_group: 2,
        };
        let groups = split_into_row_groups(&table, &config).expect("split failed");

        // Predicate: temp in [24.0, 30.0]
        let pred = Predicate::FloatRange("temp".to_string(), 24.0, 30.0);

        // Only group 1 (rows 2-3 with temp [18.0, 25.0]) could match
        let matching: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| pred.could_match_row_group(g))
            .map(|(i, _)| i)
            .collect();

        // Group 0: temp [20.0, 22.5] -> max 22.5 < 24.0 -> skip
        // Group 1: temp [18.0, 25.0] -> max 25.0 >= 24.0 -> include
        // Group 2: temp [19.5] -> max 19.5 < 24.0 -> skip
        assert_eq!(matching, vec![1]);
    }

    #[test]
    fn test_predicate_evaluate_int_equals() {
        let table = make_test_table();
        let pred = Predicate::IntEquals("id".to_string(), 3);
        let mask = pred.evaluate(&table).expect("eval failed");
        assert_eq!(mask, vec![false, false, true, false, false]);
    }

    #[test]
    fn test_predicate_evaluate_str_equals() {
        let table = make_test_table();
        let pred = Predicate::StrEquals("city".to_string(), "Tokyo".to_string());
        let mask = pred.evaluate(&table).expect("eval failed");
        assert_eq!(mask, vec![true, false, true, false, false]);
    }

    #[test]
    fn test_predicate_and() {
        let table = make_test_table();
        let pred = Predicate::And(vec![
            Predicate::FloatRange("temp".to_string(), 19.0, 23.0),
            Predicate::BoolEquals("active".to_string(), true),
        ]);
        let mask = pred.evaluate(&table).expect("eval failed");
        // temp in [19..23] => rows 0(20),1(22.5),4(19.5)
        // active=true      => rows 0,1,3
        // AND               => rows 0,1
        assert_eq!(mask, vec![true, true, false, false, false]);
    }

    #[test]
    fn test_predicate_or() {
        let table = make_test_table();
        let pred = Predicate::Or(vec![
            Predicate::IntEquals("id".to_string(), 1),
            Predicate::IntEquals("id".to_string(), 5),
        ]);
        let mask = pred.evaluate(&table).expect("eval failed");
        assert_eq!(mask, vec![true, false, false, false, true]);
    }

    #[test]
    fn test_select_columns() {
        let table = make_test_table();
        let sub = select_columns(&table, &["temp", "city"]).expect("select failed");
        assert_eq!(sub.num_columns(), 2);
        assert_eq!(sub.column_names(), vec!["temp", "city"]);
    }

    #[test]
    fn test_filter_table() {
        let table = make_test_table();
        let pred = Predicate::BoolEquals("active".to_string(), true);
        let filtered = filter_table(&table, &pred).expect("filter failed");
        assert_eq!(filtered.num_rows(), 3);
        let ids = filtered.get_i64("id").expect("get id failed");
        assert_eq!(ids, &[1, 2, 4]);
    }

    #[test]
    fn test_filter_table_combined() {
        let table = make_test_table();
        // Filter: city="Tokyo" AND temp >= 18.0 (both Tokyo rows: 20.0 and 18.0)
        let pred = Predicate::And(vec![
            Predicate::StrEquals("city".to_string(), "Tokyo".to_string()),
            Predicate::FloatRange("temp".to_string(), 18.0, f64::MAX),
        ]);
        let filtered = filter_table(&table, &pred).expect("filter failed");
        assert_eq!(filtered.num_rows(), 2);
        let temps = filtered.get_f64("temp").expect("get temp failed");
        assert!((temps[0] - 20.0).abs() < 1e-10);
        assert!((temps[1] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_projection_read() {
        let dir = std::env::temp_dir().join("scirs2_col_proj_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("proj.scircol");

        let table = make_test_table();
        super::super::writer::write_columnar(&path, &table).expect("write failed");

        let sub = read_columnar_with_columns(&path, &["id", "active"]).expect("read failed");
        assert_eq!(sub.num_columns(), 2);
        assert_eq!(sub.column_names(), vec!["id", "active"]);
        assert_eq!(sub.num_rows(), 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_empty_table_stats() {
        let table = ColumnarTable::new();
        let stats = TableStats::from_table(&table);
        assert_eq!(stats.num_rows, 0);
        assert!(stats.columns.is_empty());
    }

    #[test]
    fn test_empty_column_stats() {
        let col = Column::float64("empty", Vec::new());
        let stats = ColumnStats::from_column(&col);
        assert_eq!(stats.count, 0);
        assert!(stats.min.is_none());
        assert!(stats.max.is_none());
        assert!(stats.sum.is_none());
    }

    #[test]
    fn test_could_contain_value() {
        let col = Column::float64("x", vec![10.0, 20.0, 30.0]);
        let stats = ColumnStats::from_column(&col);
        assert!(stats.could_contain_value(15.0));
        assert!(stats.could_contain_value(10.0));
        assert!(stats.could_contain_value(30.0));
        assert!(!stats.could_contain_value(5.0));
        assert!(!stats.could_contain_value(35.0));
    }

    #[test]
    fn test_row_groups_empty_table() {
        let table = ColumnarTable::new();
        let groups =
            split_into_row_groups(&table, &RowGroupConfig::default()).expect("split failed");
        assert!(groups.is_empty());
    }
}
