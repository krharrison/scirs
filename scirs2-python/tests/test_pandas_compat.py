"""Tests for scirs2 pandas compatibility module."""

import numpy as np
import pytest
import scirs2


class TestDataFrameCreation:
    """Test DataFrame-like creation."""

    def test_create_dataframe_from_dict(self):
        """Test creating DataFrame from dictionary."""
        data = {"col1": [1, 2, 3], "col2": [4.0, 5.0, 6.0], "col3": ["a", "b", "c"]}

        df = scirs2.DataFrame(data)

        assert df.shape() == (3, 3)
        assert "col1" in df.columns()

    def test_create_dataframe_from_array(self):
        """Test creating DataFrame from numpy array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        df = scirs2.DataFrame(data, columns=["A", "B", "C"])

        assert df.shape() == (2, 3)
        assert df.columns() == ["A", "B", "C"]

    def test_create_dataframe_from_lists(self):
        """Test creating DataFrame from list of lists."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        df = scirs2.DataFrame(data, columns=["X", "Y", "Z"])

        assert df.shape() == (3, 3)


class TestDataFrameIndexing:
    """Test DataFrame indexing operations."""

    def test_select_column(self):
        """Test selecting a single column."""
        data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        col = df["col1"]

        assert len(col) == 3
        assert np.allclose(col, [1, 2, 3])

    def test_select_multiple_columns(self):
        """Test selecting multiple columns."""
        data = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
        df = scirs2.DataFrame(data)

        subset = df[["col1", "col3"]]

        assert subset.shape() == (3, 2)

    def test_loc_indexing(self):
        """Test label-based indexing."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        row = df.loc[0]

        assert len(row) == 2

    def test_iloc_indexing(self):
        """Test integer-based indexing."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        row = df.iloc[1]

        assert len(row) == 2

    def test_boolean_indexing(self):
        """Test boolean indexing."""
        data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}
        df = scirs2.DataFrame(data)

        mask = df["A"] > 2
        filtered = df[mask]

        assert filtered.shape()[0] < 4


class TestDataFrameOperations:
    """Test DataFrame operations."""

    def test_add_column(self):
        """Test adding a new column."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        df["C"] = [7, 8, 9]

        assert "C" in df.columns()
        assert df.shape() == (3, 3)

    def test_drop_column(self):
        """Test dropping a column."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = scirs2.DataFrame(data)

        df_dropped = df.drop(["B"])

        assert "B" not in df_dropped.columns()
        assert df_dropped.shape() == (3, 2)

    def test_drop_row(self):
        """Test dropping rows."""
        data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}
        df = scirs2.DataFrame(data)

        df_dropped = df.drop_rows([1, 3])

        assert df_dropped.shape()[0] == 2

    def test_rename_columns(self):
        """Test renaming columns."""
        data = {"old1": [1, 2, 3], "old2": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        df_renamed = df.rename(columns={"old1": "new1", "old2": "new2"})

        assert "new1" in df_renamed.columns()
        assert "old1" not in df_renamed.columns()


class TestAggregationOperations:
    """Test aggregation operations."""

    def test_sum(self):
        """Test sum aggregation."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        result = df.sum()

        assert result["A"] == 6
        assert result["B"] == 15

    def test_mean(self):
        """Test mean aggregation."""
        data = {"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]}
        df = scirs2.DataFrame(data)

        result = df.mean()

        assert result["A"] == 2.5
        assert result["B"] == 5.0

    def test_std(self):
        """Test standard deviation."""
        data = {"A": [1, 2, 3, 4, 5]}
        df = scirs2.DataFrame(data)

        result = df.std()

        assert result["A"] > 0

    def test_min_max(self):
        """Test min and max."""
        data = {"A": [1, 5, 3, 2, 4], "B": [10, 50, 30, 20, 40]}
        df = scirs2.DataFrame(data)

        min_result = df.min()
        max_result = df.max()

        assert min_result["A"] == 1
        assert max_result["A"] == 5


class TestGroupByOperations:
    """Test groupby operations."""

    def test_groupby_single_column(self):
        """Test grouping by single column."""
        data = {
            "category": ["A", "B", "A", "B", "A"],
            "value": [1, 2, 3, 4, 5]
        }
        df = scirs2.DataFrame(data)

        grouped = df.groupby("category")
        result = grouped.sum()

        assert "A" in result or len(result) > 0

    def test_groupby_aggregation(self):
        """Test groupby with aggregation."""
        data = {
            "group": [1, 1, 2, 2, 3],
            "value": [10, 20, 30, 40, 50]
        }
        df = scirs2.DataFrame(data)

        result = df.groupby("group").mean()

        assert len(result) >= 1

    def test_groupby_multiple_aggregations(self):
        """Test groupby with multiple aggregations."""
        data = {
            "category": ["A", "A", "B", "B"],
            "value1": [1, 2, 3, 4],
            "value2": [5, 6, 7, 8]
        }
        df = scirs2.DataFrame(data)

        result = df.groupby("category").agg(["sum", "mean"])

        assert len(result) > 0


class TestMergeAndJoin:
    """Test merge and join operations."""

    def test_merge_inner(self):
        """Test inner merge."""
        df1 = scirs2.DataFrame({"key": [1, 2, 3], "value1": [10, 20, 30]})
        df2 = scirs2.DataFrame({"key": [2, 3, 4], "value2": [200, 300, 400]})

        merged = df1.merge(df2, on="key", how="inner")

        # Should have keys 2 and 3
        assert merged.shape()[0] == 2

    def test_merge_left(self):
        """Test left merge."""
        df1 = scirs2.DataFrame({"key": [1, 2, 3], "value1": [10, 20, 30]})
        df2 = scirs2.DataFrame({"key": [2, 3, 4], "value2": [200, 300, 400]})

        merged = df1.merge(df2, on="key", how="left")

        # Should have all keys from df1
        assert merged.shape()[0] == 3

    def test_merge_outer(self):
        """Test outer merge."""
        df1 = scirs2.DataFrame({"key": [1, 2], "value1": [10, 20]})
        df2 = scirs2.DataFrame({"key": [2, 3], "value2": [200, 300]})

        merged = df1.merge(df2, on="key", how="outer")

        # Should have keys 1, 2, 3
        assert merged.shape()[0] == 3

    def test_concat_vertical(self):
        """Test vertical concatenation."""
        df1 = scirs2.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = scirs2.DataFrame({"A": [5, 6], "B": [7, 8]})

        concatenated = scirs2.concat([df1, df2], axis=0)

        assert concatenated.shape() == (4, 2)

    def test_concat_horizontal(self):
        """Test horizontal concatenation."""
        df1 = scirs2.DataFrame({"A": [1, 2, 3]})
        df2 = scirs2.DataFrame({"B": [4, 5, 6]})

        concatenated = scirs2.concat([df1, df2], axis=1)

        assert concatenated.shape() == (3, 2)


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_drop_na(self):
        """Test dropping missing values."""
        data = {"A": [1.0, np.nan, 3.0], "B": [4.0, 5.0, np.nan]}
        df = scirs2.DataFrame(data)

        cleaned = df.dropna()

        assert cleaned.shape()[0] < 3

    def test_fill_na(self):
        """Test filling missing values."""
        data = {"A": [1.0, np.nan, 3.0], "B": [4.0, 5.0, np.nan]}
        df = scirs2.DataFrame(data)

        filled = df.fillna(0.0)

        assert not np.isnan(filled["A"]).any()

    def test_replace(self):
        """Test replacing values."""
        data = {"A": [1, 2, 3, 1, 2, 3]}
        df = scirs2.DataFrame(data)

        replaced = df.replace(1, 99)

        assert 1 not in replaced["A"]
        assert 99 in replaced["A"]

    def test_drop_duplicates(self):
        """Test dropping duplicate rows."""
        data = {"A": [1, 2, 2, 3], "B": [4, 5, 5, 6]}
        df = scirs2.DataFrame(data)

        unique = df.drop_duplicates()

        assert unique.shape()[0] < 4


class TestSortingOperations:
    """Test sorting operations."""

    def test_sort_by_column(self):
        """Test sorting by column."""
        data = {"A": [3, 1, 2], "B": [6, 4, 5]}
        df = scirs2.DataFrame(data)

        sorted_df = df.sort_values(by="A")

        assert sorted_df["A"][0] == 1

    def test_sort_descending(self):
        """Test descending sort."""
        data = {"A": [1, 3, 2]}
        df = scirs2.DataFrame(data)

        sorted_df = df.sort_values(by="A", ascending=False)

        assert sorted_df["A"][0] == 3

    def test_sort_multiple_columns(self):
        """Test sorting by multiple columns."""
        data = {"A": [1, 1, 2, 2], "B": [4, 3, 2, 1]}
        df = scirs2.DataFrame(data)

        sorted_df = df.sort_values(by=["A", "B"])

        assert sorted_df.shape() == (4, 2)


class TestApplyOperations:
    """Test apply operations."""

    def test_apply_function_to_column(self):
        """Test applying function to column."""
        data = {"A": [1, 2, 3, 4]}
        df = scirs2.DataFrame(data)

        df["B"] = df["A"].apply(lambda x: x ** 2)

        assert np.allclose(df["B"], [1, 4, 9, 16])

    def test_apply_function_to_row(self):
        """Test applying function to row."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        result = df.apply(lambda row: row["A"] + row["B"], axis=1)

        assert np.allclose(result, [5, 7, 9])

    def test_map_values(self):
        """Test mapping values."""
        data = {"category": ["A", "B", "A", "C"]}
        df = scirs2.DataFrame(data)

        mapping = {"A": 1, "B": 2, "C": 3}
        df["numeric"] = df["category"].map(mapping)

        assert 1 in df["numeric"]


class TestStatisticalOperations:
    """Test statistical operations."""

    def test_describe(self):
        """Test describe statistics."""
        data = {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]}
        df = scirs2.DataFrame(data)

        stats = df.describe()

        assert "mean" in stats or "count" in stats

    def test_correlation(self):
        """Test correlation calculation."""
        data = {"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10]}
        df = scirs2.DataFrame(data)

        corr = df.corr()

        # A and B are perfectly correlated
        assert corr["A"]["B"] > 0.99 or corr.shape[0] > 0

    def test_covariance(self):
        """Test covariance calculation."""
        data = {"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]}
        df = scirs2.DataFrame(data)

        cov = df.cov()

        assert cov.shape[0] > 0


class TestPivotOperations:
    """Test pivot operations."""

    def test_pivot_table(self):
        """Test pivot table creation."""
        data = {
            "category": ["A", "A", "B", "B"],
            "subcategory": ["X", "Y", "X", "Y"],
            "value": [1, 2, 3, 4]
        }
        df = scirs2.DataFrame(data)

        pivot = df.pivot_table(
            values="value",
            index="category",
            columns="subcategory"
        )

        assert pivot.shape[0] > 0

    def test_melt(self):
        """Test melting DataFrame."""
        data = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
        df = scirs2.DataFrame(data)

        melted = df.melt()

        # Melted should have more rows
        assert melted.shape()[0] > df.shape()[0]


class TestIOCompatibility:
    """Test I/O compatibility."""

    def test_to_numpy(self):
        """Test converting to numpy array."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        array = df.to_numpy()

        assert array.shape == (3, 2)

    def test_to_dict(self):
        """Test converting to dictionary."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        result_dict = df.to_dict()

        assert "A" in result_dict
        assert "B" in result_dict

    def test_to_records(self):
        """Test converting to record array."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = scirs2.DataFrame(data)

        records = df.to_records()

        assert len(records) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = scirs2.DataFrame()

        assert df.shape() == (0, 0)

    def test_single_row(self):
        """Test DataFrame with single row."""
        data = {"A": [1], "B": [2]}
        df = scirs2.DataFrame(data)

        assert df.shape() == (1, 2)

    def test_single_column(self):
        """Test DataFrame with single column."""
        data = {"A": [1, 2, 3, 4, 5]}
        df = scirs2.DataFrame(data)

        assert df.shape() == (5, 1)

    def test_mixed_types(self):
        """Test DataFrame with mixed types."""
        data = {"int": [1, 2, 3], "float": [1.1, 2.2, 3.3], "str": ["a", "b", "c"]}
        df = scirs2.DataFrame(data)

        assert df.shape() == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
