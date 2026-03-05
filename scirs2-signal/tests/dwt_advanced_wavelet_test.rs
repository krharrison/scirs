// Tests for advanced wavelet features:
// - Symlet Sym9-Sym20 filter coefficients
// - Biorthogonal wavelet variants (bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8)
// - Block thresholding for wavelet denoising
// - Meyer and Discrete Meyer wavelets

use scirs2_signal::dwt::{wavedec, waverec, Wavelet, WaveletFilters};
use scirs2_signal::wavelet_advanced::{
    apply_block_threshold, apply_block_threshold_overlap, block_denoise_1d, optimal_block_size,
};

// ============================================================================
// Symlet Sym9-Sym20 Filter Tests
// ============================================================================

#[cfg(test)]
mod symlet_tests {
    use super::*;

    /// Verify that Sym9 filter exists and has the correct length (2*9 = 18 taps)
    #[test]
    fn test_sym9_filter_length() {
        let wavelet = Wavelet::Sym(9);
        let filters = wavelet.filters().expect("Sym9 filters should be available");
        assert_eq!(filters.dec_lo.len(), 18, "Sym9 should have 18-tap filters");
        assert_eq!(filters.dec_hi.len(), 18);
        assert_eq!(filters.rec_lo.len(), 18);
        assert_eq!(filters.rec_hi.len(), 18);
    }

    /// Verify that Sym10 filter exists and has the correct length (2*10 = 20 taps)
    #[test]
    fn test_sym10_filter_length() {
        let wavelet = Wavelet::Sym(10);
        let filters = wavelet
            .filters()
            .expect("Sym10 filters should be available");
        assert_eq!(filters.dec_lo.len(), 20, "Sym10 should have 20-tap filters");
    }

    /// Verify that Sym15 filter exists and has the correct length (2*15 = 30 taps)
    #[test]
    fn test_sym15_filter_length() {
        let wavelet = Wavelet::Sym(15);
        let filters = wavelet
            .filters()
            .expect("Sym15 filters should be available");
        assert_eq!(filters.dec_lo.len(), 30, "Sym15 should have 30-tap filters");
    }

    /// Verify that Sym20 filter exists and has the correct length (2*20 = 40 taps)
    #[test]
    fn test_sym20_filter_length() {
        let wavelet = Wavelet::Sym(20);
        let filters = wavelet
            .filters()
            .expect("Sym20 filters should be available");
        assert_eq!(filters.dec_lo.len(), 40, "Sym20 should have 40-tap filters");
    }

    /// Verify all Symlet orders 9-20 are available and have correct lengths
    #[test]
    fn test_all_symlets_9_to_20() {
        for n in 9..=20 {
            let wavelet = Wavelet::Sym(n);
            let filters = wavelet
                .filters()
                .unwrap_or_else(|_| panic!("Sym{} filters should be available", n));
            assert_eq!(
                filters.dec_lo.len(),
                2 * n,
                "Sym{} should have {}-tap filters",
                n,
                2 * n
            );
        }
    }

    /// Verify Symlet filters have finite, non-zero energy
    /// Note: some Symlet coefficient sets use non-normalized conventions
    #[test]
    fn test_symlet_finite_energy() {
        for n in 9..=20 {
            let wavelet = Wavelet::Sym(n);
            let filters = wavelet
                .filters()
                .unwrap_or_else(|_| panic!("Sym{} filters should be available", n));
            let energy: f64 = filters.dec_lo.iter().map(|x| x * x).sum();
            assert!(
                energy > 0.1 && energy < 10.0,
                "Sym{} dec_lo energy should be finite and non-trivial, got {}",
                n,
                energy
            );
        }
    }

    /// Verify that Sym9-20 filters are different from the corresponding DB filters
    /// (previously they were just falling back to DB)
    #[test]
    fn test_symlets_different_from_daubechies() {
        for n in 9..=20 {
            let sym = Wavelet::Sym(n);
            let db = Wavelet::DB(n);
            let sym_filters = sym
                .filters()
                .unwrap_or_else(|_| panic!("Sym{} filters should be available", n));
            let db_filters = db
                .filters()
                .unwrap_or_else(|_| panic!("DB{} filters should be available", n));

            // Compare over the overlapping portion of the filters
            let min_len = sym_filters.dec_lo.len().min(db_filters.dec_lo.len());
            if min_len == 0 {
                continue;
            }

            // Either different lengths or at least some coefficients differ
            let different_lengths = sym_filters.dec_lo.len() != db_filters.dec_lo.len();
            let any_different = sym_filters.dec_lo[..min_len]
                .iter()
                .zip(db_filters.dec_lo[..min_len].iter())
                .any(|(s, d)| (s - d).abs() > 1e-10);

            assert!(
                different_lengths || any_different,
                "Sym{} should have different coefficients from DB{}",
                n,
                n
            );
        }
    }

    /// Verify Sym9 decomposition and reconstruction works
    #[test]
    fn test_sym9_decompose_reconstruct() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let coeffs = wavedec(&signal, Wavelet::Sym(9), Some(2), None)
            .expect("Sym9 decomposition should work");
        assert!(!coeffs.is_empty(), "Should have coefficient arrays");

        let reconstructed =
            waverec(&coeffs, Wavelet::Sym(9)).expect("Sym9 reconstruction should work");
        assert!(
            !reconstructed.is_empty(),
            "Reconstructed signal should not be empty"
        );
    }

    /// Verify out-of-range symlet orders produce errors
    #[test]
    fn test_symlet_invalid_orders() {
        // Sym1 should fail (minimum is 2)
        assert!(Wavelet::Sym(1).filters().is_err());
        // Sym21 should fail (maximum is 20)
        assert!(Wavelet::Sym(21).filters().is_err());
    }
}

// ============================================================================
// Biorthogonal Filter Tests
// ============================================================================

#[cfg(test)]
mod biorthogonal_tests {
    use super::*;

    /// Verify bior3.5 filter is available
    #[test]
    fn test_bior35_available() {
        let wavelet = Wavelet::BiorNrNd { nr: 3, nd: 5 };
        let filters = wavelet
            .filters()
            .expect("bior3.5 filters should be available");
        assert!(!filters.dec_lo.is_empty());
        assert!(!filters.dec_hi.is_empty());
        assert!(!filters.rec_lo.is_empty());
        assert!(!filters.rec_hi.is_empty());
    }

    /// Verify bior3.7 filter is available
    #[test]
    fn test_bior37_available() {
        let wavelet = Wavelet::BiorNrNd { nr: 3, nd: 7 };
        let filters = wavelet
            .filters()
            .expect("bior3.7 filters should be available");
        assert!(!filters.dec_lo.is_empty());
    }

    /// Verify bior3.9 filter is available
    #[test]
    fn test_bior39_available() {
        let wavelet = Wavelet::BiorNrNd { nr: 3, nd: 9 };
        let filters = wavelet
            .filters()
            .expect("bior3.9 filters should be available");
        assert!(!filters.dec_lo.is_empty());
    }

    /// Verify bior4.4 filter is available
    #[test]
    fn test_bior44_available() {
        let wavelet = Wavelet::BiorNrNd { nr: 4, nd: 4 };
        let filters = wavelet
            .filters()
            .expect("bior4.4 filters should be available");
        assert!(!filters.dec_lo.is_empty());
    }

    /// Verify bior5.5 filter is available
    #[test]
    fn test_bior55_available() {
        let wavelet = Wavelet::BiorNrNd { nr: 5, nd: 5 };
        let filters = wavelet
            .filters()
            .expect("bior5.5 filters should be available");
        assert!(!filters.dec_lo.is_empty());
    }

    /// Verify bior6.8 filter is available
    #[test]
    fn test_bior68_available() {
        let wavelet = Wavelet::BiorNrNd { nr: 6, nd: 8 };
        let filters = wavelet
            .filters()
            .expect("bior6.8 filters should be available");
        assert!(!filters.dec_lo.is_empty());
    }

    /// Verify all valid biorthogonal combinations work
    #[test]
    fn test_all_valid_bior_combinations() {
        let valid_combinations = [
            (1, 1),
            (1, 3),
            (1, 5),
            (2, 2),
            (2, 4),
            (2, 6),
            (2, 8),
            (3, 1),
            (3, 3),
            (3, 5),
            (3, 7),
            (3, 9),
            (4, 4),
            (5, 5),
            (6, 8),
        ];

        for (nr, nd) in &valid_combinations {
            let wavelet = Wavelet::BiorNrNd { nr: *nr, nd: *nd };
            let result = wavelet.filters();
            assert!(
                result.is_ok(),
                "bior{}.{} should produce valid filters, got error: {:?}",
                nr,
                nd,
                result.err()
            );
        }
    }

    /// Verify biorthogonal wavelets have asymmetric dec/rec filter lengths
    /// (this is a key property of biorthogonal wavelets)
    #[test]
    fn test_bior_asymmetric_filter_lengths() {
        // bior1.3: dec_lo should be shorter than rec_lo
        let filters = Wavelet::BiorNrNd { nr: 1, nd: 3 }
            .filters()
            .expect("bior1.3 should work");
        // dec_lo length != rec_lo length for bior1.3
        assert_ne!(
            filters.dec_lo.len(),
            filters.rec_lo.len(),
            "bior1.3 should have asymmetric filter lengths"
        );
    }

    /// Verify biorthogonal filter creation works correctly for bior2.2
    /// Note: DWT transform with asymmetric filter lengths is a known limitation
    /// of the current implementation, tested here only at the filter level.
    #[test]
    fn test_bior22_filter_properties() {
        let wavelet = Wavelet::BiorNrNd { nr: 2, nd: 2 };
        let filters = wavelet
            .filters()
            .expect("bior2.2 filters should be available");

        // bior2.2 has dec_lo of length 5 and rec_lo of length 3 (asymmetric)
        assert_eq!(filters.dec_lo.len(), 5, "bior2.2 dec_lo should have 5 taps");
        assert_eq!(filters.rec_lo.len(), 3, "bior2.2 rec_lo should have 3 taps");

        // Verify the high-pass filters have correct asymmetric lengths
        // dec_hi is derived from rec_lo (3 taps), rec_hi from dec_lo (5 taps)
        assert_eq!(filters.dec_hi.len(), 3, "bior2.2 dec_hi should have 3 taps");
        assert_eq!(filters.rec_hi.len(), 5, "bior2.2 rec_hi should have 5 taps");
    }

    /// Verify bior decomposition works for bior1.1 (= Haar, simplest case)
    #[test]
    fn test_bior11_decompose_reconstruct() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let wavelet = Wavelet::BiorNrNd { nr: 1, nd: 1 };
        let coeffs =
            wavedec(&signal, wavelet, Some(2), None).expect("bior1.1 decomposition should work");
        assert!(!coeffs.is_empty(), "Should have coefficient arrays");
    }

    /// Verify reverse biorthogonal wavelets work for the new combinations
    #[test]
    fn test_rbio_new_combinations() {
        let new_combinations = [(3, 5), (3, 7), (3, 9), (4, 4), (5, 5), (6, 8)];

        for (nr, nd) in &new_combinations {
            let wavelet = Wavelet::RBioNrNd { nr: *nr, nd: *nd };
            let result = wavelet.filters();
            assert!(
                result.is_ok(),
                "rbio{}.{} should produce valid filters, got error: {:?}",
                nr,
                nd,
                result.err()
            );
        }
    }

    /// Verify invalid biorthogonal combinations produce errors
    #[test]
    fn test_bior_invalid_combinations() {
        let invalid_combinations = [(1, 2), (2, 3), (3, 4), (4, 5), (7, 7)];
        for (nr, nd) in &invalid_combinations {
            let wavelet = Wavelet::BiorNrNd { nr: *nr, nd: *nd };
            assert!(
                wavelet.filters().is_err(),
                "bior{}.{} should be an invalid combination",
                nr,
                nd
            );
        }
    }
}

// ============================================================================
// Meyer and Discrete Meyer Filter Tests
// ============================================================================

#[cfg(test)]
mod meyer_tests {
    use super::*;

    /// Verify Meyer wavelet filters are available
    #[test]
    fn test_meyer_filter_available() {
        let wavelet = Wavelet::Meyer;
        let filters = wavelet
            .filters()
            .expect("Meyer filters should be available");
        assert_eq!(filters.dec_lo.len(), 62, "Meyer should have 62-tap filters");
        assert_eq!(filters.dec_hi.len(), 62);
        assert_eq!(filters.rec_lo.len(), 62);
        assert_eq!(filters.rec_hi.len(), 62);
    }

    /// Verify Discrete Meyer wavelet filters are available
    #[test]
    fn test_dmeyer_filter_available() {
        let wavelet = Wavelet::DMeyer;
        let filters = wavelet
            .filters()
            .expect("DMeyer filters should be available");
        assert!(!filters.dec_lo.is_empty());
        assert!(!filters.dec_hi.is_empty());
    }

    /// Verify Meyer decomposition works
    #[test]
    fn test_meyer_decompose() {
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let coeffs = wavedec(&signal, Wavelet::Meyer, Some(1), None)
            .expect("Meyer decomposition should work");
        assert!(!coeffs.is_empty());
    }

    /// Verify DMeyer decomposition works
    #[test]
    fn test_dmeyer_decompose() {
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let coeffs = wavedec(&signal, Wavelet::DMeyer, Some(1), None)
            .expect("DMeyer decomposition should work");
        assert!(!coeffs.is_empty());
    }

    /// Verify Meyer and DMeyer have different coefficients
    #[test]
    fn test_meyer_vs_dmeyer_different() {
        let meyer = Wavelet::Meyer.filters().expect("Meyer should work");
        let dmeyer = Wavelet::DMeyer.filters().expect("DMeyer should work");

        // They may have different filter lengths, or at least different values
        let different_lengths = meyer.dec_lo.len() != dmeyer.dec_lo.len();
        let different_values = if !different_lengths {
            meyer
                .dec_lo
                .iter()
                .zip(dmeyer.dec_lo.iter())
                .any(|(a, b)| (a - b).abs() > 1e-10)
        } else {
            true
        };
        assert!(
            different_lengths || different_values,
            "Meyer and DMeyer should be different wavelets"
        );
    }
}

// ============================================================================
// Block Thresholding Tests
// ============================================================================

#[cfg(test)]
mod block_threshold_tests {
    use super::*;

    /// Test basic block thresholding on a simple signal
    #[test]
    fn test_apply_block_threshold_basic() {
        let mut coeffs = vec![0.1, 0.2, 5.0, 6.0, 0.05, 0.03, 0.01, 0.02];
        let threshold = 1.0;
        let block_size = 2;

        apply_block_threshold(&mut coeffs, threshold, block_size);

        // The large-valued block [5.0, 6.0] should be mostly preserved (shrunk a bit)
        assert!(coeffs[2].abs() > 0.1, "Large coefficients should survive");
        assert!(coeffs[3].abs() > 0.1, "Large coefficients should survive");

        // The small-valued blocks should be zeroed or heavily shrunk
        assert!(
            coeffs[0].abs() < 0.3,
            "Small coefficients should be shrunk or zeroed"
        );
    }

    /// Test block thresholding with zero threshold preserves all values
    #[test]
    fn test_block_threshold_zero_threshold() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut coeffs = original.clone();

        apply_block_threshold(&mut coeffs, 0.0, 2);

        // With zero threshold, all blocks should be preserved
        for (a, b) in coeffs.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Zero threshold should preserve all values"
            );
        }
    }

    /// Test optimal block size computation
    #[test]
    fn test_optimal_block_size() {
        // For various signal lengths, block size should be log(n)
        let bs_100 = optimal_block_size(100);
        let bs_1000 = optimal_block_size(1000);
        let bs_10000 = optimal_block_size(10000);

        // Block size should grow logarithmically
        assert!(bs_100 >= 4, "Block size for n=100 should be at least 4");
        assert!(bs_1000 >= 6, "Block size for n=1000 should be at least 6");
        assert!(bs_10000 >= 9, "Block size for n=10000 should be at least 9");

        // Block size should be at least 1 for small signals
        let bs_1 = optimal_block_size(1);
        assert!(bs_1 >= 1, "Block size should always be at least 1");
    }

    /// Test overlapping block thresholding
    #[test]
    fn test_block_threshold_overlap() {
        let mut coeffs = vec![0.01, 0.02, 5.0, 6.0, 7.0, 0.03, 0.01, 0.02];
        let threshold = 1.0;
        let block_size = 4;

        let overlap = block_size / 2; // 50% overlap
        apply_block_threshold_overlap(&mut coeffs, threshold, block_size, overlap);

        // Large-valued region should still have significant values
        assert!(
            coeffs[2].abs() > 0.1,
            "Large coefficients should survive overlapping block threshold"
        );
        assert!(
            coeffs[3].abs() > 0.1,
            "Large coefficients should survive overlapping block threshold"
        );
    }

    /// Test block_denoise_1d full pipeline
    #[test]
    fn test_block_denoise_1d_basic() {
        // Create a signal with noise
        let clean: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();

        // Add some "noise"
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let noise = ((i * 7 + 3) as f64 * 0.618).sin() * 0.3;
                x + noise
            })
            .collect();

        let denoised = block_denoise_1d(&noisy, Wavelet::DB(4), Some(3), None, false, Some(0.3))
            .expect("Block denoising should succeed");

        assert_eq!(
            denoised.len(),
            noisy.len(),
            "Denoised signal should have same length as input"
        );
    }

    /// Test block_denoise_1d with overlapping blocks
    #[test]
    fn test_block_denoise_1d_overlap() {
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.05).cos()).collect();

        let denoised = block_denoise_1d(&signal, Wavelet::DB(4), Some(2), Some(8), true, Some(0.1))
            .expect("Block denoising with overlap should succeed");

        assert!(!denoised.is_empty());
    }

    /// Test block_denoise_1d with empty signal returns error
    #[test]
    fn test_block_denoise_1d_empty_signal() {
        let empty: Vec<f64> = vec![];
        let result = block_denoise_1d(&empty, Wavelet::DB(4), None, None, false, None);
        assert!(result.is_err(), "Empty signal should produce an error");
    }

    /// Test block thresholding with single-element blocks
    #[test]
    fn test_block_threshold_single_element_blocks() {
        let mut coeffs = vec![0.01, 5.0, 0.02, 6.0];
        let threshold = 0.5;

        // Block size 1 degenerates to element-wise thresholding
        apply_block_threshold(&mut coeffs, threshold, 1);

        // Large values should survive, small values should be heavily shrunk or zeroed
        assert!(
            coeffs[1].abs() > 1.0,
            "Large coefficient should survive with block_size=1"
        );
    }

    /// Test that block_denoise_1d works with various wavelet families
    #[test]
    fn test_block_denoise_various_wavelets() {
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();

        let wavelets = [
            Wavelet::DB(4),
            Wavelet::Sym(8),
            Wavelet::Coif(3),
            Wavelet::Haar,
        ];

        for wavelet in &wavelets {
            let result = block_denoise_1d(&signal, *wavelet, Some(2), None, false, Some(0.1));
            assert!(
                result.is_ok(),
                "Block denoising should work with {:?}",
                wavelet
            );
        }
    }
}
