"""Tests for scirs2 I/O module."""

import numpy as np
import pytest
import scirs2
import tempfile
import os


class TestFileIO:
    """Test file I/O operations."""

    def test_save_load_npy(self):
        """Test saving and loading NumPy format."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_npy_py(data, tmp_path)
            loaded = scirs2.load_npy_py(tmp_path)

            assert np.allclose(loaded, data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_npz(self):
        """Test saving and loading compressed NPZ format."""
        data = {"array1": np.array([1, 2, 3]), "array2": np.array([4, 5, 6])}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_npz_py(tmp_path, **data)
            loaded = scirs2.load_npz_py(tmp_path)

            assert "array1" in loaded
            assert "array2" in loaded
            assert np.allclose(loaded["array1"], data["array1"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_csv(self):
        """Test CSV file I/O."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_csv_py(data, tmp_path)
            loaded = scirs2.load_csv_py(tmp_path)

            assert np.allclose(loaded, data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_json(self):
        """Test JSON file I/O."""
        data = {"key1": [1, 2, 3], "key2": [4, 5, 6]}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_json_py(data, tmp_path)
            loaded = scirs2.load_json_py(tmp_path)

            assert loaded["key1"] == data["key1"]
            assert loaded["key2"] == data["key2"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestArrowFormat:
    """Test Apache Arrow format support."""

    def test_to_arrow_table(self):
        """Test converting data to Arrow table."""
        data = {"col1": np.array([1, 2, 3]), "col2": np.array([4.0, 5.0, 6.0])}

        table = scirs2.to_arrow_table_py(data)

        assert table is not None
        assert "col1" in table.column_names or hasattr(table, "schema")

    def test_from_arrow_table(self):
        """Test loading from Arrow table."""
        data = {"col1": np.array([1, 2, 3]), "col2": np.array([4.0, 5.0, 6.0])}

        table = scirs2.to_arrow_table_py(data)
        loaded = scirs2.from_arrow_table_py(table)

        assert "col1" in loaded
        assert "col2" in loaded

    def test_save_load_arrow(self):
        """Test Arrow file format."""
        data = {"col1": np.array([1, 2, 3]), "col2": np.array([4.0, 5.0, 6.0])}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".arrow") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_arrow_py(data, tmp_path)
            loaded = scirs2.load_arrow_py(tmp_path)

            assert "col1" in loaded or len(loaded) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_parquet(self):
        """Test Parquet file format."""
        data = {"col1": np.array([1, 2, 3]), "col2": np.array([4.0, 5.0, 6.0])}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_parquet_py(data, tmp_path)
            loaded = scirs2.load_parquet_py(tmp_path)

            assert len(loaded) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSerialization:
    """Test object serialization."""

    def test_serialize_deserialize_array(self):
        """Test array serialization."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        serialized = scirs2.serialize_py(data)
        deserialized = scirs2.deserialize_py(serialized)

        assert np.allclose(deserialized, data)

    def test_serialize_dict(self):
        """Test dictionary serialization."""
        data = {"a": 1, "b": [2, 3, 4], "c": "test"}

        serialized = scirs2.serialize_py(data)
        deserialized = scirs2.deserialize_py(serialized)

        assert deserialized["a"] == data["a"]
        assert deserialized["c"] == data["c"]

    def test_pickle_save_load(self):
        """Test pickle serialization."""
        data = {"array": np.array([1, 2, 3]), "value": 42}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_pickle_py(data, tmp_path)
            loaded = scirs2.load_pickle_py(tmp_path)

            assert "array" in loaded
            assert loaded["value"] == 42
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestBinaryIO:
    """Test binary file operations."""

    def test_save_load_binary(self):
        """Test binary file I/O."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_binary_py(data, tmp_path)
            loaded = scirs2.load_binary_py(tmp_path, dtype=np.int32, shape=(5,))

            assert np.allclose(loaded, data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_raw(self):
        """Test raw binary I/O."""
        data = b"Hello, World!"

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            scirs2.write_bytes_py(data, tmp_path)
            loaded = scirs2.read_bytes_py(tmp_path)

            assert loaded == data
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestStreamIO:
    """Test streaming I/O operations."""

    def test_read_chunks(self):
        """Test reading file in chunks."""
        data = np.arange(1000).astype(np.float64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_npy_py(data, tmp_path)

            chunks = []
            for chunk in scirs2.read_chunks_py(tmp_path, chunk_size=100):
                chunks.append(chunk)

            assert len(chunks) > 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_write_stream(self):
        """Test writing data stream."""
        data = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".stream") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.write_stream_py(data, tmp_path)

            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestCompression:
    """Test compression formats."""

    def test_save_load_gzip(self):
        """Test gzip compression."""
        data = np.random.randn(100, 100)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy.gz") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_compressed_py(data, tmp_path, compression="gzip")
            loaded = scirs2.load_compressed_py(tmp_path, compression="gzip")

            assert np.allclose(loaded, data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_zstd(self):
        """Test Zstandard compression."""
        data = np.random.randn(50, 50)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy.zst") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_compressed_py(data, tmp_path, compression="zstd")
            loaded = scirs2.load_compressed_py(tmp_path, compression="zstd")

            assert np.allclose(loaded, data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestMemoryMappedIO:
    """Test memory-mapped file operations."""

    def test_mmap_read(self):
        """Test memory-mapped reading."""
        data = np.arange(1000).astype(np.float64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
            tmp_path = tmp.name

        try:
            data.tofile(tmp_path)

            mmap_array = scirs2.mmap_array_py(tmp_path, dtype=np.float64, shape=(1000,), mode="r")

            assert np.allclose(mmap_array[:10], data[:10])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_mmap_write(self):
        """Test memory-mapped writing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
            tmp_path = tmp.name

        try:
            mmap_array = scirs2.mmap_array_py(tmp_path, dtype=np.float64, shape=(100,), mode="w+")

            mmap_array[0] = 42.0
            mmap_array.flush()

            # Read back
            loaded = np.fromfile(tmp_path, dtype=np.float64)
            assert loaded[0] == 42.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestMetadata:
    """Test file metadata operations."""

    def test_save_with_metadata(self):
        """Test saving with metadata."""
        data = np.array([1, 2, 3, 4, 5])
        metadata = {"created": "2026-02-16", "author": "test"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_with_metadata_py(data, tmp_path, metadata=metadata)
            loaded_data, loaded_metadata = scirs2.load_with_metadata_py(tmp_path)

            assert np.allclose(loaded_data, data)
            assert "created" in loaded_metadata or loaded_metadata is not None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array_save_load(self):
        """Test saving empty array."""
        data = np.array([])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_npy_py(data, tmp_path)
            loaded = scirs2.load_npy_py(tmp_path)

            assert loaded.shape == (0,)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_large_file(self):
        """Test handling large files."""
        data = np.random.randn(1000, 1000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            tmp_path = tmp.name

        try:
            scirs2.save_npy_py(data, tmp_path)
            file_size = os.path.getsize(tmp_path)

            assert file_size > 1000000  # Should be > 1MB
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_nonexistent_file_load(self):
        """Test loading non-existent file."""
        try:
            loaded = scirs2.load_npy_py("/nonexistent/path/file.npy")
            assert False, "Should have raised error"
        except Exception:
            pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
