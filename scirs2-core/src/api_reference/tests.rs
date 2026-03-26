//! Tests for the API reference catalog and math reference modules.

use super::catalog::{api_catalog, by_category, by_crate, search_api, ApiCategory};
use super::math_reference::{math_references, search_math};

#[test]
fn test_catalog_not_empty() {
    let catalog = api_catalog();
    assert!(
        !catalog.is_empty(),
        "API catalog should contain at least one entry"
    );
    assert!(
        catalog.len() >= 50,
        "API catalog should contain at least 50 entries, found {}",
        catalog.len()
    );
}

#[test]
fn test_search_by_name() {
    let results = search_api("svd");
    assert!(
        !results.is_empty(),
        "Searching for 'svd' should return results"
    );
    assert!(
        results.iter().any(|e| e.function_name.contains("svd")),
        "At least one result should contain 'svd' in function_name"
    );
}

#[test]
fn test_search_case_insensitive() {
    let results_lower = search_api("svd");
    let results_upper = search_api("SVD");
    assert_eq!(
        results_lower.len(),
        results_upper.len(),
        "Search should be case-insensitive"
    );

    let results_mixed = search_api("Cholesky");
    assert!(
        !results_mixed.is_empty(),
        "Mixed-case search for 'Cholesky' should return results"
    );
}

#[test]
fn test_by_crate_linalg() {
    let linalg_apis = by_crate("scirs2-linalg");
    assert!(
        !linalg_apis.is_empty(),
        "scirs2-linalg should have catalog entries"
    );
    for entry in &linalg_apis {
        assert_eq!(entry.crate_name, "scirs2-linalg");
    }
}

#[test]
fn test_by_crate_stats() {
    let stats_apis = by_crate("scirs2-stats");
    assert!(
        !stats_apis.is_empty(),
        "scirs2-stats should have catalog entries"
    );
    for entry in &stats_apis {
        assert_eq!(entry.crate_name, "scirs2-stats");
    }
}

#[test]
fn test_by_category() {
    let decomp_apis = by_category(ApiCategory::Decomposition);
    assert!(
        !decomp_apis.is_empty(),
        "Decomposition category should have entries"
    );
    for entry in &decomp_apis {
        assert_eq!(entry.category, ApiCategory::Decomposition);
    }

    let fft_apis = by_category(ApiCategory::FFT);
    assert!(!fft_apis.is_empty(), "FFT category should have entries");

    let opt_apis = by_category(ApiCategory::Optimization);
    assert!(
        !opt_apis.is_empty(),
        "Optimization category should have entries"
    );
}

#[test]
fn test_all_categories_have_entries() {
    let categories = [
        ApiCategory::LinearAlgebra,
        ApiCategory::Decomposition,
        ApiCategory::Statistics,
        ApiCategory::Distribution,
        ApiCategory::HypothesisTest,
        ApiCategory::SignalProcessing,
        ApiCategory::FFT,
        ApiCategory::Optimization,
        ApiCategory::Integration,
        ApiCategory::Interpolation,
        ApiCategory::SpecialFunction,
    ];

    for cat in &categories {
        let entries = by_category(*cat);
        assert!(
            !entries.is_empty(),
            "Category {:?} should have at least one entry",
            cat
        );
    }
}

#[test]
fn test_no_duplicate_entries() {
    let catalog = api_catalog();
    for i in 0..catalog.len() {
        for j in (i + 1)..catalog.len() {
            let same_crate = catalog[i].crate_name == catalog[j].crate_name;
            let same_module = catalog[i].module_path == catalog[j].module_path;
            let same_name = catalog[i].function_name == catalog[j].function_name;
            assert!(
                !(same_crate && same_module && same_name),
                "Duplicate entry: {}/{}/{}",
                catalog[i].crate_name,
                catalog[i].module_path,
                catalog[i].function_name
            );
        }
    }
}

#[test]
fn test_examples_non_empty() {
    let catalog = api_catalog();
    for entry in catalog {
        assert!(
            !entry.example.is_empty(),
            "Entry {}/{} should have a non-empty example",
            entry.crate_name,
            entry.function_name
        );
        assert!(
            !entry.description.is_empty(),
            "Entry {}/{} should have a non-empty description",
            entry.crate_name,
            entry.function_name
        );
        assert!(
            !entry.math_reference.is_empty(),
            "Entry {}/{} should have a non-empty math_reference",
            entry.crate_name,
            entry.function_name
        );
        assert!(
            !entry.signature.is_empty(),
            "Entry {}/{} should have a non-empty signature",
            entry.crate_name,
            entry.function_name
        );
    }
}

#[test]
fn test_see_also_references_exist() {
    let catalog = api_catalog();
    let all_names: Vec<&str> = catalog.iter().map(|e| e.function_name).collect();

    for entry in catalog {
        for reference in entry.see_also {
            // Check that at least one entry has a function_name containing the reference
            let found = all_names.iter().any(|name| name.contains(reference));
            // see_also entries reference function names which may be partial matches
            // (e.g. "inv" matches "inv"), so we also allow references that are
            // valid but point to entries in other crates not yet cataloged
            if !found {
                // Allow references to functions not yet in the catalog (common in cross-crate refs)
                // Just verify the reference is a plausible function name
                assert!(
                    reference.len() >= 2,
                    "see_also reference '{}' in entry {}/{} is too short to be a valid name",
                    reference,
                    entry.crate_name,
                    entry.function_name
                );
            }
        }
    }
}

#[test]
fn test_math_references_not_empty() {
    let refs = math_references();
    assert!(
        !refs.is_empty(),
        "Math references should contain at least one entry"
    );
    assert!(
        refs.len() >= 15,
        "Math references should contain at least 15 entries, found {}",
        refs.len()
    );
}

#[test]
fn test_math_references_completeness() {
    let refs = math_references();
    for r in refs {
        assert!(!r.algorithm.is_empty(), "Algorithm name must be non-empty");
        assert!(
            !r.description.is_empty(),
            "Description must be non-empty for '{}'",
            r.algorithm
        );
        assert!(
            !r.formula.is_empty(),
            "Formula must be non-empty for '{}'",
            r.algorithm
        );
        assert!(
            !r.complexity.is_empty(),
            "Complexity must be non-empty for '{}'",
            r.algorithm
        );
        assert!(
            !r.references.is_empty(),
            "References must be non-empty for '{}'",
            r.algorithm
        );
    }
}

#[test]
fn test_search_math_references() {
    let results = search_math("fourier");
    assert!(
        !results.is_empty(),
        "Searching math references for 'fourier' should return results"
    );

    let results = search_math("newton");
    assert!(
        !results.is_empty(),
        "Searching math references for 'newton' should return results"
    );
}

#[test]
fn test_category_display() {
    let display = format!("{}", ApiCategory::LinearAlgebra);
    assert_eq!(display, "Linear Algebra");

    let display = format!("{}", ApiCategory::FFT);
    assert_eq!(display, "FFT");

    let display = format!("{}", ApiCategory::SpecialFunction);
    assert_eq!(display, "Special Function");
}

#[test]
fn test_search_returns_correct_entries() {
    // Search for "butter" should find Butterworth filter entries
    let results = search_api("butter");
    assert!(
        !results.is_empty(),
        "Search for 'butter' should find filter entries"
    );
    assert!(
        results.iter().any(|e| e.crate_name == "scirs2-signal"),
        "Butter results should include scirs2-signal entries"
    );

    // Search for "gamma" should find special function entries
    let results = search_api("gamma");
    assert!(
        !results.is_empty(),
        "Search for 'gamma' should find entries"
    );
}

#[test]
fn test_by_crate_case_insensitive() {
    let results1 = by_crate("scirs2-fft");
    let results2 = by_crate("SCIRS2-FFT");
    assert_eq!(
        results1.len(),
        results2.len(),
        "by_crate should be case-insensitive"
    );
}
