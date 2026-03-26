//! Fuzz target: Zarr metadata parsing.
//!
//! Verifies that:
//! 1. `ArrayMetadataV2::from_json` never panics on arbitrary bytes.
//! 2. `ArrayMetadataV3::from_json` never panics on arbitrary bytes.
//! 3. `ConsolidatedMetadata::from_json` never panics on arbitrary bytes.
//! 4. Round-trips (from_json → to_json) on valid metadata never panic.
//! 5. `GroupMetadataV3::from_json` never panics.
//!
//! All errors must be reported as `Result::Err`, never as panics.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // ── Zarr v2 array metadata ────────────────────────────────────────────────
    {
        let result = scirs2_io::zarr::ArrayMetadataV2::from_json(data);
        if let Ok(meta) = result {
            // Round-trip: serialise back to JSON — must not panic.
            let _ = meta.to_json();
            // Probe derived helpers.
            let _ = meta.data_type();
        }
    }

    // ── Zarr v3 array metadata ────────────────────────────────────────────────
    {
        let result = scirs2_io::zarr::ArrayMetadataV3::from_json(data);
        if let Ok(meta) = result {
            let _ = meta.to_json();
            let _ = meta.data_type_parsed();
            let _ = meta.chunk_shape();
        }
    }

    // ── Zarr v2 consolidated metadata ────────────────────────────────────────
    {
        let result = scirs2_io::zarr::ConsolidatedMetadata::from_json(data);
        if let Ok(meta) = result {
            let _ = meta.to_json();
        }
    }

    // ── GroupMetadataV3 via serde_json ────────────────────────────────────────
    // GroupMetadataV3 is serde-derived; parse it directly via serde_json.
    {
        if let Ok(meta) =
            serde_json::from_slice::<scirs2_io::zarr::GroupMetadataV3>(data)
        {
            // Round-trip: re-serialise to ensure no panics in derived Debug/Display.
            let _ = serde_json::to_vec(&meta);
        }
    }

    // ── CodecMetadata JSON parse ──────────────────────────────────────────────
    // Exercise codec metadata deserialisation on arbitrary UTF-8.
    if let Ok(s) = std::str::from_utf8(data) {
        if let Ok(codec) =
            serde_json::from_str::<scirs2_io::zarr::CodecMetadata>(s)
        {
            drop(codec);
        }
    }
});
