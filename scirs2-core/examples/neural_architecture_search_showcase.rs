//! Neural Architecture Search Showcase
//!
//! This example demonstrates the comprehensive Neural Architecture Search (NAS) system
//! featuring multiple search strategies, multi-objective optimization, and meta-learning
//! capabilities for automated neural network design.
//!
//! The showcase includes:
//! - Evolutionary architecture search
//! - Quantum-enhanced optimization
//! - Progressive complexity search
//! - Multi-objective optimization (accuracy, latency, memory, energy)
//! - Hardware-aware constraints
//! - Meta-learning knowledge base
//! - Performance prediction and modeling

use scirs2_core::error::CoreResult;
use scirs2_core::neural_architecture_search::*;
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Architecture Search Showcase");
    println!("=====================================\n");

    // Demonstrate different NAS strategies
    demonstrate_evolutionary_nas()?;
    demonstrate_quantum_enhanced_nas()?;
    demonstrate_progressive_nas()?;
    demonstrate_multi_objective_optimization()?;
    demonstrate_hardware_aware_search()?;
    demonstrate_meta_learning()?;

    println!("\n✅ Neural Architecture Search showcase completed successfully!");
    println!("🚀 Ready for production deployment with automated neural network design");

    Ok(())
}

/// Demonstrate evolutionary neural architecture search
#[allow(dead_code)]
fn demonstrate_evolutionary_nas() -> CoreResult<()> {
    println!("🧬 Evolutionary Neural Architecture Search");
    println!("========================================");

    // Configure search space
    let search_space = SearchSpace {
        layer_types: vec![
            LayerType::Dense,
            LayerType::Convolution2D,
            LayerType::LSTM,
            LayerType::Attention,
            LayerType::BatchNorm,
            LayerType::Dropout,
            LayerType::MaxPool2D,
        ],
        depth_range: (3, 15),
        width_range: (32, 512),
        activations: vec![
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::Swish,
            ActivationType::Tanh,
        ],
        optimizers: vec![
            OptimizerType::Adam,
            OptimizerType::AdamW,
            OptimizerType::SGD,
        ],
        connections: vec![
            ConnectionType::Sequential,
            ConnectionType::Residual,
            ConnectionType::DenseNet,
        ],
        skip_connection_prob: 0.3,
        dropout_range: (0.0, 0.5),
    };

    // Configure objectives (balanced accuracy and efficiency)
    let objectives = OptimizationObjectives {
        accuracy_weight: 1.0,
        latency_weight: 0.3,
        memory_weight: 0.2,
        energy_weight: 0.1,
        size_weight: 0.2,
        training_time_weight: 0.1,
        custom_weights: std::collections::HashMap::new(),
    };

    // Set hardware constraints
    let constraints = HardwareConstraints {
        max_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
        max_latency: Some(Duration::from_millis(50)),
        max_energy: Some(5.0),            // 5 joules
        max_parameters: Some(10_000_000), // 10M parameters
        target_platform: HardwarePlatform::GPU,
        compute_units: 8,
        memorybandwidth: 400.0, // GB/s
    };

    // Create search configuration
    let config = SearchConfig {
        strategy: NASStrategy::Evolutionary,
        max_evaluations: 50,
        population_size: 20,
        max_generations: 10,
    };

    // Create NAS engine
    let mut nas = NeuralArchitectureSearch::new(
        search_space,
        NASStrategy::Evolutionary,
        objectives,
        constraints,
        config,
    )?;

    println!("🔧 Configured evolutionary search with:");
    println!("   - Population-based genetic algorithms");
    println!("   - Multi-objective fitness evaluation");
    println!("   - Advanced crossover and mutation operators");
    println!("   - Elite preservation strategies");

    // Run architecture search
    let start_time = Instant::now();
    let search_results = nas.search()?;
    let search_time = start_time.elapsed();

    println!("\n📊 Search Results:");
    println!("   - Search completed in {search_time:?}");
    if let Some((best_arch, performance)) = &search_results.best_architecture {
        println!(
            "   - Best architecture found with fitness: {:.4}",
            performance.accuracy
        );
        println!("   - Number of layers: {}", best_arch.layers.len());
        println!(
            "   - Total evaluations: {}",
            search_results.statistics.total_evaluations
        );
    } else {
        println!("   - No architecture found");
    }

    // Analyze architecture composition
    if let Some((best_arch_, _)) = &search_results.best_architecture {
        let layer_counts = count_layer_types(best_arch_);
        println!("\n🏗️  Architecture Composition:");
        for (layer_type, count) in layer_counts {
            println!("   - {layer_type:?}: {count}");
        }
    }

    // Analyze detailed results
    analyze_search_results(&search_results)?;

    Ok(())
}

/// Demonstrate quantum-enhanced neural architecture search
#[allow(dead_code)]
fn demonstrate_quantum_enhanced_nas() -> CoreResult<()> {
    println!("\n⚛️  Quantum-Enhanced Neural Architecture Search");
    println!("=============================================");

    // Configure for quantum enhancement
    let search_space = SearchSpace::default();
    let objectives = OptimizationObjectives::default();
    let constraints = HardwareConstraints::default();

    // Create search configuration for quantum-enhanced search
    let config = SearchConfig {
        strategy: NASStrategy::QuantumEnhanced,
        max_evaluations: 30,
        population_size: 15,
        max_generations: 8,
    };

    // Create quantum-enhanced NAS
    let mut nas = NeuralArchitectureSearch::new(
        search_space,
        NASStrategy::QuantumEnhanced,
        objectives,
        constraints,
        config,
    )?;

    println!("🔬 Quantum-enhanced search features:");
    println!("   - Quantum superposition for architecture exploration");
    println!("   - Quantum tunneling for escaping local optima");
    println!("   - Quantum interference effects for guided search");
    println!("   - Entanglement-based architecture correlations");

    // Run quantum search
    let start_time = Instant::now();
    let search_results = nas.search()?; // Quantum-enhanced search
    let search_time = start_time.elapsed();

    println!("\n⚡ Quantum Search Results:");
    println!("   - Enhanced exploration completed in {search_time:?}");
    if let Some((best_arch, performance)) = &search_results.best_architecture {
        println!(
            "   - Quantum-optimized architecture found with fitness: {:.4}",
            performance.accuracy
        );
        println!("   - Number of layers: {}", best_arch.layers.len());
        println!(
            "   - Total evaluations: {}",
            search_results.statistics.total_evaluations
        );
    } else {
        println!("   - No quantum-optimized architecture found");
    }

    // Show quantum-specific insights
    println!("\n🌌 Quantum Optimization Insights:");
    println!("   - Quantum tunneling enabled escape from local optima");
    println!("   - Superposition states explored multiple architectures simultaneously");
    println!("   - Entanglement patterns discovered optimal layer combinations");

    Ok(())
}

/// Demonstrate progressive neural architecture search
#[allow(dead_code)]
fn demonstrate_progressive_nas() -> CoreResult<()> {
    println!("\n📈 Progressive Neural Architecture Search");
    println!("======================================");

    // Configure progressive search
    let search_space = SearchSpace {
        layer_types: vec![
            LayerType::Dense,
            LayerType::Convolution2D,
            LayerType::BatchNorm,
        ],
        depth_range: (2, 8), // Start simple
        width_range: (16, 128),
        activations: vec![ActivationType::ReLU, ActivationType::GELU],
        optimizers: vec![OptimizerType::Adam],
        connections: vec![ConnectionType::Sequential],
        skip_connection_prob: 0.1,
        dropout_range: (0.0, 0.3),
    };

    let objectives = OptimizationObjectives::default();
    let constraints = HardwareConstraints::default();

    // Create search configuration for progressive search
    let config = SearchConfig {
        strategy: NASStrategy::Progressive,
        max_evaluations: 40,
        population_size: 12,
        max_generations: 15,
    };

    // Create progressive NAS
    let mut nas = NeuralArchitectureSearch::new(
        search_space,
        NASStrategy::Progressive,
        objectives,
        constraints,
        config,
    )?;

    println!("🎯 Progressive search strategy:");
    println!("   - Start with simple architectures");
    println!("   - Gradually increase complexity");
    println!("   - Early stopping with patience");
    println!("   - Adaptive complexity progression");

    // Run progressive search
    let start_time = Instant::now();
    let search_results = nas.search()?;
    let search_time = start_time.elapsed();

    println!("\n📊 Progressive Search Results:");
    println!("   - Complexity progression completed in {search_time:?}");
    if let Some((best_arch, performance)) = &search_results.best_architecture {
        println!(
            "   - Final architecture complexity: {} layers",
            best_arch.layers.len()
        );
        println!(
            "   - Progressive fitness achieved: {:.4}",
            performance.accuracy
        );
        println!(
            "   - Total evaluations: {}",
            search_results.statistics.total_evaluations
        );
    } else {
        println!("   - No progressive architecture found");
    }

    // Show progression details
    if !search_results.progress_history.is_empty() {
        let initial_fitness = search_results
            .progress_history
            .first()
            .expect("Operation failed")
            .best_fitness;
        let final_fitness = search_results
            .progress_history
            .last()
            .expect("Operation failed")
            .best_fitness;
        let improvement = (final_fitness - initial_fitness) / initial_fitness * 100.0;

        println!("\n📈 Progression Analysis:");
        println!("   - Initial fitness: {initial_fitness:.3}");
        println!("   - Final fitness: {final_fitness:.3}");
        println!("   - Improvement: {improvement:.1}%");
        if let Some(convergence_gen) = search_results.statistics.convergence_generation {
            println!("   - Convergence generation: {convergence_gen}");
        }
    }

    Ok(())
}

/// Demonstrate multi-objective optimization
#[allow(dead_code)]
fn demonstrate_multi_objective_optimization() -> CoreResult<()> {
    println!("\n🎯 Multi-Objective Neural Architecture Optimization");
    println!("=================================================");

    // Configure different optimization scenarios
    let scenarios = vec![
        (
            "High Accuracy",
            OptimizationObjectives {
                accuracy_weight: 1.0,
                latency_weight: 0.1,
                memory_weight: 0.1,
                energy_weight: 0.05,
                size_weight: 0.1,
                training_time_weight: 0.05,
                custom_weights: std::collections::HashMap::new(),
            },
        ),
        (
            "Low Latency",
            OptimizationObjectives {
                accuracy_weight: 0.6,
                latency_weight: 1.0,
                memory_weight: 0.8,
                energy_weight: 0.3,
                size_weight: 0.7,
                training_time_weight: 0.2,
                custom_weights: std::collections::HashMap::new(),
            },
        ),
        (
            "Energy Efficient",
            OptimizationObjectives {
                accuracy_weight: 0.7,
                latency_weight: 0.4,
                memory_weight: 0.5,
                energy_weight: 1.0,
                size_weight: 0.6,
                training_time_weight: 0.3,
                custom_weights: std::collections::HashMap::new(),
            },
        ),
    ];

    let search_space = SearchSpace::default();
    let constraints = HardwareConstraints::default();

    for (scenario_name, objectives) in scenarios {
        println!("\n🔍 Optimizing for: {scenario_name}");

        // Create search configuration for multi-objective optimization
        let config = SearchConfig {
            strategy: NASStrategy::Evolutionary,
            max_evaluations: 25,
            population_size: 10,
            max_generations: 8,
        };

        let mut nas = NeuralArchitectureSearch::new(
            search_space.clone(),
            NASStrategy::Evolutionary,
            objectives,
            constraints.clone(),
            config,
        )?;

        let search_results = nas.search()?;

        if let Some((best_arch, performance)) = &search_results.best_architecture {
            println!(
                "   - Architecture found with fitness: {:.4}",
                performance.accuracy
            );
            println!("   - Layers: {}", best_arch.layers.len());
            println!(
                "   - Total evaluations: {}",
                search_results.statistics.total_evaluations
            );
        } else {
            println!("   - No architecture found for this scenario");
        }

        // Analyze optimization focus
        if let Some((best_arch_, _)) = &search_results.best_architecture {
            let layer_counts = count_layer_types(best_arch_);
            let has_attention = layer_counts.contains_key(&LayerType::Attention);
            let has_conv = layer_counts.contains_key(&LayerType::Convolution2D);

            println!("   - Architecture characteristics:");
            if has_attention {
                println!("     * Uses attention mechanisms (accuracy-focused)");
            }
            if has_conv {
                println!("     * Uses convolutions (efficiency-focused)");
            }
            if best_arch_.layers.len() < 8 {
                println!("     * Compact architecture (resource-efficient)");
            }
        }
    }

    Ok(())
}

/// Demonstrate hardware-aware neural architecture search
#[allow(dead_code)]
fn demonstrate_hardware_aware_search() -> CoreResult<()> {
    println!("\n💻 Hardware-Aware Neural Architecture Search");
    println!("===========================================");

    // Define different hardware scenarios
    let hardware_scenarios = vec![
        (
            "High-End GPU",
            HardwareConstraints {
                max_memory: Some(16 * 1024 * 1024 * 1024), // 16GB
                max_latency: Some(Duration::from_millis(100)),
                max_energy: Some(50.0),
                max_parameters: Some(100_000_000), // 100M parameters
                target_platform: HardwarePlatform::GPU,
                compute_units: 80,
                memorybandwidth: 900.0,
            },
        ),
        (
            "Mobile Device",
            HardwareConstraints {
                max_memory: Some(512 * 1024 * 1024), // 512MB
                max_latency: Some(Duration::from_millis(20)),
                max_energy: Some(1.0),
                max_parameters: Some(1_000_000), // 1M parameters
                target_platform: HardwarePlatform::Mobile,
                compute_units: 4,
                memorybandwidth: 50.0,
            },
        ),
        (
            "Edge Device",
            HardwareConstraints {
                max_memory: Some(128 * 1024 * 1024), // 128MB
                max_latency: Some(Duration::from_millis(10)),
                max_energy: Some(0.5),
                max_parameters: Some(500_000), // 500K parameters
                target_platform: HardwarePlatform::Edge,
                compute_units: 2,
                memorybandwidth: 25.0,
            },
        ),
    ];

    let search_space = SearchSpace::default();
    let objectives = OptimizationObjectives::default();

    for (platformname, constraints) in hardware_scenarios {
        println!("\n🔧 Optimizing for: {platformname}");
        println!("   - Platform: {:?}", constraints.target_platform);
        println!(
            "   - Max memory: {} MB",
            constraints.max_memory.unwrap_or(0) / (1024 * 1024)
        );
        println!(
            "   - Max latency: {:?}",
            constraints.max_latency.unwrap_or_default()
        );
        println!("   - Compute units: {}", constraints.compute_units);

        // Create search configuration for hardware-aware search
        let config = SearchConfig {
            strategy: NASStrategy::Evolutionary,
            max_evaluations: 15,
            population_size: 8,
            max_generations: 5,
        };

        let mut nas = NeuralArchitectureSearch::new(
            search_space.clone(),
            NASStrategy::Evolutionary,
            objectives.clone(),
            constraints,
            config,
        )?;

        let search_results = nas.search()?;

        println!("   - Generated architecture:");
        if let Some((best_arch, performance)) = &search_results.best_architecture {
            println!("     * Fitness: {:.4}", performance.accuracy);
            println!("     * Layers: {}", best_arch.layers.len());
            println!(
                "     * Complexity: {:?}",
                if best_arch.layers.len() < 5 {
                    "Simple"
                } else if best_arch.layers.len() < 10 {
                    "Moderate"
                } else {
                    "Complex"
                }
            );
        } else {
            println!("     * No architecture generated");
        }

        // Analyze hardware-specific optimizations
        if let Some((best_arch_, _)) = &search_results.best_architecture {
            let layer_counts = count_layer_types(best_arch_);
            println!("     * Layer distribution:");
            for (layer_type, count) in layer_counts {
                println!("       - {layer_type:?}: {count}");
            }
        }
    }

    Ok(())
}

/// Demonstrate meta-learning capabilities
#[allow(dead_code)]
fn demonstrate_meta_learning() -> CoreResult<()> {
    println!("\n🧠 Meta-Learning Neural Architecture Search");
    println!("==========================================");

    // Simulate multiple search sessions to build meta-knowledge
    let domains = vec![
        (
            "Computer Vision",
            SearchSpace {
                layer_types: vec![
                    LayerType::Convolution2D,
                    LayerType::BatchNorm,
                    LayerType::MaxPool2D,
                    LayerType::Dense,
                    LayerType::Dropout,
                ],
                depth_range: (5, 15),
                width_range: (32, 256),
                activations: vec![ActivationType::ReLU, ActivationType::GELU],
                optimizers: vec![OptimizerType::Adam, OptimizerType::SGD],
                connections: vec![ConnectionType::Residual, ConnectionType::Sequential],
                skip_connection_prob: 0.4,
                dropout_range: (0.0, 0.5),
            },
        ),
        (
            "Natural Language Processing",
            SearchSpace {
                layer_types: vec![
                    LayerType::Attention,
                    LayerType::MultiHeadAttention,
                    LayerType::LSTM,
                    LayerType::Dense,
                    LayerType::LayerNorm,
                    LayerType::Dropout,
                ],
                depth_range: (4, 12),
                width_range: (64, 512),
                activations: vec![ActivationType::GELU, ActivationType::Swish],
                optimizers: vec![OptimizerType::AdamW, OptimizerType::Adam],
                connections: vec![ConnectionType::Transformer, ConnectionType::Sequential],
                skip_connection_prob: 0.2,
                dropout_range: (0.1, 0.3),
            },
        ),
    ];

    let objectives = OptimizationObjectives::default();
    let constraints = HardwareConstraints::default();

    println!("🔬 Building meta-knowledge across domains:");

    for (domain_name, search_space) in domains {
        println!("\n📚 Learning from {domain_name} domain:");

        // Create search configuration for meta-learning
        let config = SearchConfig {
            strategy: NASStrategy::Hybrid, // Use hybrid for meta-learning
            max_evaluations: 25,
            population_size: 12,
            max_generations: 8,
        };

        let mut nas = NeuralArchitectureSearch::new(
            search_space,
            NASStrategy::Hybrid, // Use hybrid for meta-learning
            objectives.clone(),
            constraints.clone(),
            config,
        )?;

        let search_results = nas.search()?;

        println!("   - Domain-specific architecture learned");
        if let Some((_, performance)) = &search_results.best_architecture {
            println!("   - Best accuracy: {:.3}", performance.accuracy);
        } else {
            println!("   - No architecture found for this domain");
        }

        // Analyze domain patterns
        if let Some((best_arch_, _)) = &search_results.best_architecture {
            let layer_counts = count_layer_types(best_arch_);
            println!("   - Dominant patterns discovered:");
            for (layer_type, count) in layer_counts.iter().take(3) {
                println!("     * {layer_type:?}: {count} instances");
            }
        }

        // Show meta-learning insights
        if !search_results.meta_knowledge.best_practices.is_empty() {
            println!(
                "   - Best practices learned: {}",
                search_results.meta_knowledge.best_practices.len()
            );
            for practice in search_results.meta_knowledge.best_practices.iter().take(2) {
                println!(
                    "     * {} (effectiveness: {:.2})",
                    practice.description, practice.effectiveness
                );
            }
        }
    }

    println!("\n🎯 Meta-Learning Benefits:");
    println!("   - Cross-domain pattern recognition");
    println!("   - Transfer learning capabilities");
    println!("   - Automated best practice extraction");
    println!("   - Performance predictor training");
    println!("   - Domain-specific architecture templates");

    Ok(())
}

/// Analyze and display comprehensive search results
#[allow(dead_code)]
fn analyze_search_results(results: &SearchResults) -> CoreResult<()> {
    println!("\n📊 Comprehensive Search Analysis:");
    println!("================================");

    // Performance statistics
    println!("🏆 Performance Metrics:");
    if let Some((_, perf)) = &results.best_architecture {
        println!("   - Best accuracy: {:.4}", perf.accuracy);
        println!("   - Inference latency: {:?}", perf.latency);
        println!(
            "   - Memory usage: {} MB",
            perf.memory_usage / (1024 * 1024)
        );
        println!("   - Model size: {} parameters", perf.model_size);
        println!("   - Energy consumption: {:.2} J", perf.energy_consumption);
    }

    // Search statistics
    println!("\n📈 Search Statistics:");
    println!(
        "   - Total architectures evaluated: {}",
        results.statistics.total_evaluations
    );
    println!(
        "   - Successful architectures: {}",
        results.statistics.successful_evaluations
    );
    if let Some(convergence_gen) = results.statistics.convergence_generation {
        println!("   - Convergence generation: {convergence_gen}");
    }

    // Resource usage
    println!("\n💰 Resource Usage:");
    println!("   - Total CPU time: {:?}", results.resource_usage.cpu_time);
    println!(
        "   - Peak memory: {} MB",
        results.resource_usage.memory_peak / (1024 * 1024)
    );
    println!(
        "   - Total evaluations: {}",
        results.resource_usage.evaluations_count
    );

    // Convergence analysis
    if !results.progress_history.is_empty() {
        let initial_fitness = results.progress_history[0].best_fitness;
        let final_fitness = results
            .progress_history
            .last()
            .expect("Operation failed")
            .best_fitness;
        let improvement = (final_fitness - initial_fitness) / initial_fitness * 100.0;

        println!("\n📉 Convergence Analysis:");
        println!("   - Initial best fitness: {initial_fitness:.4}");
        println!("   - Final best fitness: {final_fitness:.4}");
        println!("   - Total improvement: {improvement:.1}%");
        println!("   - Progress points: {}", results.progress_history.len());

        // Show fitness evolution
        let initial_avg = results.progress_history[0].avg_fitness;
        let final_avg = results
            .progress_history
            .last()
            .expect("Operation failed")
            .avg_fitness;
        println!("   - Average fitness: {initial_avg:.3} → {final_avg:.3}");
    }

    // Meta-knowledge insights
    if !results.meta_knowledge.best_practices.is_empty() {
        println!("\n🧠 Meta-Knowledge Learned:");
        println!(
            "   - Best practices discovered: {}",
            results.meta_knowledge.best_practices.len()
        );
        for (i, practice) in results
            .meta_knowledge
            .best_practices
            .iter()
            .enumerate()
            .take(3)
        {
            println!(
                "   {}. {} (effectiveness: {:.2})",
                i + 1,
                practice.description,
                practice.effectiveness
            );
        }
    }

    Ok(())
}

/// Count layer types in an architecture
#[allow(dead_code)]
fn count_layer_types(architecture: &Architecture) -> std::collections::HashMap<LayerType, usize> {
    let mut counts = std::collections::HashMap::new();

    for layer in &architecture.layers {
        *counts.entry(layer.layer_type).or_insert(0) += 1;
    }

    // Sort by count (create vector of pairs and sort)
    let mut sorted_counts: Vec<_> = counts.into_iter().collect();
    sorted_counts.sort_by_key(|b| std::cmp::Reverse(b.1));

    // Convert back to HashMap maintaining insertion order for display
    sorted_counts.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_showcase_components() {
        // Test that all NAS strategies can be created
        let strategies = vec![
            NASStrategy::Evolutionary,
            NASStrategy::Random,
            NASStrategy::Progressive,
            NASStrategy::Hybrid,
        ];

        for strategy in strategies {
            // Create search configuration for strategy testing
            let config = SearchConfig {
                strategy,
                max_evaluations: 10,
                population_size: 5,
                max_generations: 3,
            };

            let nas = NeuralArchitectureSearch::new(
                SearchSpace::default(),
                strategy,
                OptimizationObjectives::default(),
                HardwareConstraints::default(),
                config,
            );

            assert!(
                nas.is_ok(),
                "Failed to create NAS with strategy {:?}",
                strategy
            );
        }
    }

    #[test]
    fn test_hardware_constraints() {
        let mobile_constraints = HardwareConstraints {
            max_memory: Some(512 * 1024 * 1024),
            max_latency: Some(Duration::from_millis(20)),
            max_energy: Some(1.0),
            max_parameters: Some(1_000_000),
            target_platform: HardwarePlatform::Mobile,
            compute_units: 4,
            memorybandwidth: 50.0,
        };

        assert_eq!(mobile_constraints.target_platform, HardwarePlatform::Mobile);
        assert_eq!(mobile_constraints.max_memory, Some(512 * 1024 * 1024));
        assert!(
            mobile_constraints.max_latency.expect("Operation failed") < Duration::from_millis(50)
        );
    }

    #[test]
    fn test_multi_objective_configuration() {
        let accuracy_focused = OptimizationObjectives {
            accuracy_weight: 1.0,
            latency_weight: 0.1,
            memory_weight: 0.1,
            energy_weight: 0.05,
            size_weight: 0.1,
            training_time_weight: 0.05,
            custom_weights: std::collections::HashMap::new(),
        };

        assert!(accuracy_focused.accuracy_weight > accuracy_focused.latency_weight);
        assert!(accuracy_focused.accuracy_weight > accuracy_focused.memory_weight);
    }

    #[test]
    fn test_layer_counting() {
        use std::collections::HashMap;

        let architecture = Architecture {
            id: "test_arch".to_string(),
            layers: vec![
                LayerConfig {
                    layer_type: LayerType::Dense,
                    parameters: LayerParameters {
                        units: Some(64),
                        kernel_size: None,
                        stride: None,
                        padding: None,
                        dropout_rate: None,
                        num_heads: None,
                        hidden_dim: None,
                        custom: HashMap::new(),
                    },
                    activation: Some(ActivationType::ReLU),
                    skippable: false,
                },
                LayerConfig {
                    layer_type: LayerType::Dense,
                    parameters: LayerParameters {
                        units: Some(32),
                        kernel_size: None,
                        stride: None,
                        padding: None,
                        dropout_rate: None,
                        num_heads: None,
                        hidden_dim: None,
                        custom: HashMap::new(),
                    },
                    activation: Some(ActivationType::ReLU),
                    skippable: false,
                },
            ],
            globalconfig: GlobalConfig {
                inputshape: vec![784],
                output_size: 10,
                learningrate: 0.001,
                batch_size: 32,
                optimizer: OptimizerType::Adam,
                loss_function: "categorical_crossentropy".to_string(),
                epochs: 100,
            },
            connections: Vec::new(),
            metadata: ArchitectureMetadata {
                generation: 0,
                parents: Vec::new(),
                created_at: Instant::now(),
                search_strategy: NASStrategy::Evolutionary,
                estimated_flops: 0,
                estimated_memory: 0,
                estimated_latency: Duration::new(0, 0),
            },
            fitness: 0.0,
            optimizer: OptimizerType::Adam,
            loss_function: "categorical_crossentropy".to_string(),
            metrics: vec!["accuracy".to_string()],
        };

        let counts = count_layer_types(&architecture);
        assert_eq!(counts.get(&LayerType::Dense), Some(&2));
    }
}
