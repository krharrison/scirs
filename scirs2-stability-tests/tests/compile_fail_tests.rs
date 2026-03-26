//! Compile-fail tests verifying that incorrect usage of public APIs correctly
//! fails to compile.  These act as negative-API regression tests: if they start
//! compiling successfully, a `#[non_exhaustive]` attribute (or other contract)
//! has been silently removed.

#[test]
fn compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
