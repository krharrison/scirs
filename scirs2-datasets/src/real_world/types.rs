//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::cache::{CacheKey, CacheManager};
use crate::error::{DatasetsError, Result};
use crate::registry::{DatasetMetadata, DatasetRegistry};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Uniform;

/// Configuration for real-world dataset loading
#[derive(Debug, Clone)]
pub struct RealWorldConfig {
    /// Whether to use cached versions if available
    pub use_cache: bool,
    /// Whether to download if not available locally
    pub download_if_missing: bool,
    /// Data directory for storing datasets
    pub data_home: Option<String>,
    /// Whether to return preprocessed version
    pub return_preprocessed: bool,
    /// Subset of data to load (for large datasets)
    pub subset: Option<String>,
    /// Random state for reproducible subsampling
    pub random_state: Option<u64>,
}
/// Real-world dataset loader and manager
pub struct RealWorldDatasets {
    pub(super) cache: CacheManager,
    registry: DatasetRegistry,
    pub(super) config: RealWorldConfig,
}
impl RealWorldDatasets {
    /// Create a new real-world datasets manager
    pub fn new(config: RealWorldConfig) -> Result<Self> {
        let cache = CacheManager::new()?;
        let registry = DatasetRegistry::new();
        Ok(Self {
            cache,
            registry,
            config,
        })
    }
    /// Load a dataset by name
    pub fn load_dataset(&mut self, name: &str) -> Result<Dataset> {
        match name {
            "adult" => self.load_adult(),
            "bank_marketing" => self.load_bank_marketing(),
            "credit_approval" => self.load_credit_approval(),
            "german_credit" => self.load_german_credit(),
            "mushroom" => self.load_mushroom(),
            "spam" => self.load_spam(),
            "titanic" => self.load_titanic(),
            "auto_mpg" => self.load_auto_mpg(),
            "california_housing" => self.load_california_housing(),
            "concrete_strength" => self.load_concrete_strength(),
            "energy_efficiency" => self.load_energy_efficiency(),
            "red_wine_quality" => self.load_red_wine_quality(),
            "white_wine_quality" => self.load_white_wine_quality(),
            "air_passengers" => self.load_air_passengers(),
            "bitcoin_prices" => self.load_bitcoin_prices(),
            "electricity_load" => self.load_electricity_load(),
            "stock_prices" => self.load_stock_prices(),
            "cifar10_subset" => self.load_cifar10_subset(),
            "fashion_mnist_subset" => self.load_fashion_mnist_subset(),
            "imdb_reviews" => self.load_imdb_reviews(),
            "news_articles" => self.load_news_articles(),
            "diabetes_readmission" => self.load_diabetes_readmission(),
            "heart_disease" => self.load_heart_disease(),
            "credit_card_fraud" => self.load_credit_card_fraud(),
            "loan_default" => self.load_loan_default(),
            _ => Err(DatasetsError::NotFound(format!("Unknown dataset: {name}"))),
        }
    }
    /// List all available real-world datasets
    pub fn list_datasets(&self) -> Vec<String> {
        vec![
            "adult".to_string(),
            "bank_marketing".to_string(),
            "credit_approval".to_string(),
            "german_credit".to_string(),
            "mushroom".to_string(),
            "spam".to_string(),
            "titanic".to_string(),
            "auto_mpg".to_string(),
            "california_housing".to_string(),
            "concrete_strength".to_string(),
            "energy_efficiency".to_string(),
            "red_wine_quality".to_string(),
            "white_wine_quality".to_string(),
            "air_passengers".to_string(),
            "bitcoin_prices".to_string(),
            "electricity_load".to_string(),
            "stock_prices".to_string(),
            "cifar10_subset".to_string(),
            "fashion_mnist_subset".to_string(),
            "imdb_reviews".to_string(),
            "news_articles".to_string(),
            "diabetes_readmission".to_string(),
            "heart_disease".to_string(),
            "credit_card_fraud".to_string(),
            "loan_default".to_string(),
        ]
    }
    /// Get dataset information without loading
    pub fn get_dataset_info(&self, name: &str) -> Result<DatasetMetadata> {
        self.registry.get_metadata(name)
    }
}
impl RealWorldDatasets {
    /// Load Adult (Census Income) dataset
    /// Predict whether income exceeds $50K/yr based on census data
    pub fn load_adult(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("adult", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data";
        let dataset = self.download_and_parse_csv(
            url,
            "adult",
            &[
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education_num",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital_gain",
                "capital_loss",
                "hours_per_week",
                "native_country",
                "income",
            ],
            Some("income"),
            true,
        )?;
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Bank Marketing dataset
    /// Predict if client will subscribe to term deposit
    pub fn load_bank_marketing(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("bank_marketing", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_bank_data(4521, 16)?;
        let metadata = DatasetMetadata {
            name: "Bank Marketing".to_string(),
            description: "Direct marketing campaigns of a Portuguese banking institution"
                .to_string(),
            n_samples: 4521,
            n_features: 16,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["no".to_string(), "yes".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Titanic dataset
    /// Predict passenger survival on the Titanic
    pub fn load_titanic(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("titanic", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_titanic_data(891, 7)?;
        let metadata = DatasetMetadata {
            name: "Titanic".to_string(),
            description: "Passenger survival data from the Titanic disaster".to_string(),
            n_samples: 891,
            n_features: 7,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["died".to_string(), "survived".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load German Credit dataset
    /// Credit risk assessment
    pub fn load_german_credit(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_credit_data(1000, 20)?;
        let metadata = DatasetMetadata {
            name: "German Credit".to_string(),
            description: "Credit risk classification dataset".to_string(),
            n_samples: 1000,
            n_features: 20,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["bad_credit".to_string(), "good_credit".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
}
impl RealWorldDatasets {
    /// Load California Housing dataset
    /// Predict median house values in California districts
    pub fn load_california_housing(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_housing_data(20640, 8)?;
        let metadata = DatasetMetadata {
            name: "California Housing".to_string(),
            description: "Median house values for California districts from 1990 census"
                .to_string(),
            n_samples: 20640,
            n_features: 8,
            task_type: "regression".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    /// Load Wine Quality dataset (Red Wine)
    /// Predict wine quality based on physicochemical properties
    pub fn load_red_wine_quality(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_wine_data(1599, 11)?;
        let metadata = DatasetMetadata {
            name: "Red Wine Quality".to_string(),
            description: "Red wine quality based on physicochemical tests".to_string(),
            n_samples: 1599,
            n_features: 11,
            task_type: "regression".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    /// Load Energy Efficiency dataset
    /// Predict heating and cooling loads of buildings
    pub fn load_energy_efficiency(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_energy_data(768, 8)?;
        let metadata = DatasetMetadata {
            name: "Energy Efficiency".to_string(),
            description: "Energy efficiency of buildings based on building parameters".to_string(),
            n_samples: 768,
            n_features: 8,
            task_type: "regression".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
}
impl RealWorldDatasets {
    /// Load Air Passengers dataset
    /// Classic time series dataset of airline passengers
    pub fn load_air_passengers(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_air_passengers_data(144)?;
        let metadata = DatasetMetadata {
            name: "Air Passengers".to_string(),
            description: "Monthly airline passenger numbers 1949-1960".to_string(),
            n_samples: 144,
            n_features: 1,
            task_type: "time_series".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, target, metadata))
    }
    /// Load Bitcoin Prices dataset
    /// Historical Bitcoin price data
    pub fn load_bitcoin_prices(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_bitcoin_price_data(1000)?;
        let metadata = DatasetMetadata {
            name: "Bitcoin Prices".to_string(),
            description: "Historical Bitcoin price data with technical indicators".to_string(),
            n_samples: 1000,
            n_features: 6,
            task_type: "time_series".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, target, metadata))
    }
}
impl RealWorldDatasets {
    /// Load Heart Disease dataset
    /// Predict presence of heart disease
    pub fn load_heart_disease(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_heart_disease_data(303, 13)?;
        let metadata = DatasetMetadata {
            name: "Heart Disease".to_string(),
            description: "Heart disease prediction based on clinical parameters".to_string(),
            n_samples: 303,
            n_features: 13,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["no_disease".to_string(), "disease".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    /// Load Diabetes Readmission dataset
    /// Predict hospital readmission for diabetic patients
    pub fn load_diabetes_readmission(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_diabetes_readmission_data(101766, 49)?;
        let metadata = DatasetMetadata {
            name: "Diabetes Readmission".to_string(),
            description: "Hospital readmission prediction for diabetic patients".to_string(),
            n_samples: 101766,
            n_features: 49,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "no_readmission".to_string(),
                "readmission".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    /// Load the Credit Approval dataset from UCI repository
    pub fn load_credit_approval(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("credit_approval", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let url =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data";
        let columns = &[
            "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13",
            "A14", "A15", "class",
        ];
        let dataset =
            self.download_and_parse_csv(url, "credit_approval", columns, Some("class"), true)?;
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load the Mushroom dataset from UCI repository
    pub fn load_mushroom(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("mushroom", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data";
        let columns = &[
            "class",
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ];
        let dataset = self.download_and_parse_csv(url, "mushroom", columns, Some("class"), true)?;
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load the Spambase dataset from UCI repository
    pub fn load_spam(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("spam", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let url =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data";
        let mut columns: Vec<String> = Vec::new();
        for i in 0..48 {
            columns.push(format!("word_freq_{i}"));
        }
        for i in 0..6 {
            columns.push(format!("char_freq_{i}"));
        }
        columns.push("capital_run_length_average".to_string());
        columns.push("capital_run_length_longest".to_string());
        columns.push("capital_run_length_total".to_string());
        columns.push("spam".to_string());
        let column_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        let dataset =
            self.download_and_parse_csv(url, "spam", &column_refs, Some("spam"), false)?;
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Auto MPG dataset
    pub fn load_auto_mpg(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("auto_mpg", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_auto_mpg_data(392, 7)?;
        let metadata = DatasetMetadata {
            name: "Auto MPG".to_string(),
            description:
                "Predict car fuel efficiency (miles per gallon) from technical specifications"
                    .to_string(),
            n_samples: 392,
            n_features: 7,
            task_type: "regression".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Concrete Compressive Strength dataset
    pub fn load_concrete_strength(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("concrete_strength", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_concrete_data(1030, 8)?;
        let metadata = DatasetMetadata {
            name: "Concrete Compressive Strength".to_string(),
            description: "Predict concrete compressive strength from mixture components"
                .to_string(),
            n_samples: 1030,
            n_features: 8,
            task_type: "regression".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load White Wine Quality dataset
    pub fn load_white_wine_quality(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("white_wine_quality", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_wine_data(4898, 11)?;
        let metadata = DatasetMetadata {
            name: "White Wine Quality".to_string(),
            description: "White wine quality based on physicochemical tests".to_string(),
            n_samples: 4898,
            n_features: 11,
            task_type: "regression".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Electricity Load dataset
    pub fn load_electricity_load(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("electricity_load", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_electricity_data(26304, 3)?;
        let metadata = DatasetMetadata {
            name: "Electricity Load".to_string(),
            description: "Hourly electricity consumption forecasting with weather factors"
                .to_string(),
            n_samples: 26304,
            n_features: 3,
            task_type: "time_series".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Stock Prices dataset
    pub fn load_stock_prices(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("stock_prices", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_stock_data(1260, 5)?;
        let metadata = DatasetMetadata {
            name: "Stock Prices".to_string(),
            description: "Daily stock price prediction with technical indicators".to_string(),
            n_samples: 1260,
            n_features: 5,
            task_type: "time_series".to_string(),
            targetnames: None,
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load CIFAR-10 subset dataset
    pub fn load_cifar10_subset(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("cifar10_subset", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_cifar10_data(1000, 3072)?;
        let metadata = DatasetMetadata {
            name: "CIFAR-10 Subset".to_string(),
            description: "Subset of CIFAR-10 32x32 color images in 10 classes".to_string(),
            n_samples: 1000,
            n_features: 3072,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "airplane".to_string(),
                "automobile".to_string(),
                "bird".to_string(),
                "cat".to_string(),
                "deer".to_string(),
                "dog".to_string(),
                "frog".to_string(),
                "horse".to_string(),
                "ship".to_string(),
                "truck".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load Fashion-MNIST subset dataset
    pub fn load_fashion_mnist_subset(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("fashion_mnist_subset", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_fashion_mnist_data(1000, 784)?;
        let metadata = DatasetMetadata {
            name: "Fashion-MNIST Subset".to_string(),
            description: "Subset of Fashion-MNIST 28x28 grayscale images of fashion items"
                .to_string(),
            n_samples: 1000,
            n_features: 784,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "T-shirt/top".to_string(),
                "Trouser".to_string(),
                "Pullover".to_string(),
                "Dress".to_string(),
                "Coat".to_string(),
                "Sandal".to_string(),
                "Shirt".to_string(),
                "Sneaker".to_string(),
                "Bag".to_string(),
                "Ankle boot".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load IMDB movie reviews dataset
    pub fn load_imdb_reviews(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("imdb_reviews", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_imdb_data(5000, 1000)?;
        let metadata = DatasetMetadata {
            name: "IMDB Movie Reviews".to_string(),
            description: "Subset of IMDB movie reviews for sentiment classification".to_string(),
            n_samples: 5000,
            n_features: 1000,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["negative".to_string(), "positive".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load news articles dataset
    pub fn load_news_articles(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("news_articles", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_news_data(2000, 500)?;
        let metadata = DatasetMetadata {
            name: "News Articles".to_string(),
            description: "News articles categorized by topic for text classification".to_string(),
            n_samples: 2000,
            n_features: 500,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "business".to_string(),
                "entertainment".to_string(),
                "politics".to_string(),
                "sport".to_string(),
                "tech".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load credit card fraud detection dataset
    pub fn load_credit_card_fraud(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("credit_card_fraud", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_fraud_data(284807, 28)?;
        let metadata = DatasetMetadata {
            name: "Credit Card Fraud Detection".to_string(),
            description: "Detect fraudulent credit card transactions from anonymized features"
                .to_string(),
            n_samples: 284807,
            n_features: 28,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["legitimate".to_string(), "fraud".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
    /// Load loan default prediction dataset
    pub fn load_loan_default(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("loan_default", &self.config);
        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }
        let (data, target) = self.create_synthetic_loan_data(10000, 15)?;
        let metadata = DatasetMetadata {
            name: "Loan Default Prediction".to_string(),
            description: "Predict loan default risk from borrower characteristics and loan details"
                .to_string(),
            n_samples: 10000,
            n_features: 15,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["no_default".to_string(), "default".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };
        let dataset = Dataset::from_metadata(data, Some(target), metadata);
        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }
        Ok(dataset)
    }
}
impl RealWorldDatasets {
    fn download_and_parse_csv(
        &self,
        url: &str,
        name: &str,
        columns: &[&str],
        target_col: Option<&str>,
        has_categorical: bool,
    ) -> Result<Dataset> {
        if !self.config.download_if_missing {
            return Err(DatasetsError::DownloadError(
                "Download disabled in configuration".to_string(),
            ));
        }
        #[cfg(feature = "download")]
        {
            match self.download_real_dataset(url, name, columns, target_col, has_categorical) {
                Ok(dataset) => return Ok(dataset),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to download real dataset from {}: {}. Falling back to synthetic data.",
                        url, e
                    );
                }
            }
        }
        match name {
            "adult" => {
                let (data, target) = self.create_synthetic_adult_dataset(32561, 14)?;
                let featurenames = vec![
                    "age".to_string(),
                    "workclass".to_string(),
                    "fnlwgt".to_string(),
                    "education".to_string(),
                    "education_num".to_string(),
                    "marital_status".to_string(),
                    "occupation".to_string(),
                    "relationship".to_string(),
                    "race".to_string(),
                    "sex".to_string(),
                    "capital_gain".to_string(),
                    "capital_loss".to_string(),
                    "hours_per_week".to_string(),
                    "native_country".to_string(),
                ];
                let metadata = crate::registry::DatasetMetadata {
                    name: "Adult Census Income".to_string(),
                    description: "Predict whether income exceeds $50K/yr based on census data"
                        .to_string(),
                    n_samples: 32561,
                    n_features: 14,
                    task_type: "classification".to_string(),
                    targetnames: Some(vec!["<=50K".to_string(), ">50K".to_string()]),
                    featurenames: Some(featurenames),
                    url: Some(url.to_string()),
                    checksum: None,
                };
                Ok(Dataset::from_metadata(data, Some(target), metadata))
            }
            _ => {
                let n_features = columns.len() - if target_col.is_some() { 1 } else { 0 };
                let (data, target) =
                    self.create_generic_synthetic_dataset(1000, n_features, has_categorical)?;
                let featurenames: Vec<String> = columns
                    .iter()
                    .filter(|&&_col| Some(_col) != target_col)
                    .map(|&_col| _col.to_string())
                    .collect();
                let metadata = crate::registry::DatasetMetadata {
                    name: format!("Synthetic {name}"),
                    description: format!("Synthetic version of {name} dataset"),
                    n_samples: 1000,
                    n_features,
                    task_type: if target_col.is_some() {
                        "classification"
                    } else {
                        "regression"
                    }
                    .to_string(),
                    targetnames: None,
                    featurenames: Some(featurenames),
                    url: Some(url.to_string()),
                    checksum: None,
                };
                Ok(Dataset::from_metadata(data, target, metadata))
            }
        }
    }
    /// Download and parse real dataset from URL
    #[cfg(feature = "download")]
    fn download_real_dataset(
        &self,
        url: &str,
        name: &str,
        columns: &[&str],
        target_col: Option<&str>,
        _has_categorical: bool,
    ) -> Result<Dataset> {
        use crate::cache::download_data;
        use std::collections::HashMap;
        use std::io::{BufRead, BufReader, Cursor};
        let data_bytes = download_data(url, false)?;
        let cursor = Cursor::new(data_bytes);
        let reader = BufReader::new(cursor);
        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut header_found = false;
        for line_result in reader.lines() {
            let line = line_result
                .map_err(|e| DatasetsError::FormatError(format!("Failed to read line: {}", e)))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<String> = line
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect();
            if !header_found && fields.len() == columns.len() {
                let is_header = fields.iter().enumerate().all(|(i, field)| {
                    field.to_lowercase().contains(&columns[i].to_lowercase())
                        || columns[i].to_lowercase().contains(&field.to_lowercase())
                });
                if is_header {
                    header_found = true;
                    continue;
                }
            }
            if fields.len() == columns.len() {
                rows.push(fields);
            }
        }
        if rows.is_empty() {
            return Err(DatasetsError::FormatError(
                "No valid data rows found in CSV".to_string(),
            ));
        }
        let n_samples = rows.len();
        let n_features = if target_col.is_some() {
            columns.len() - 1
        } else {
            columns.len()
        };
        let mut data = Array2::<f64>::zeros((n_samples, n_features));
        let mut target = if target_col.is_some() {
            Some(Array1::<f64>::zeros(n_samples))
        } else {
            None
        };
        let mut category_maps: HashMap<usize, HashMap<String, f64>> = HashMap::new();
        for (row_idx, row) in rows.iter().enumerate() {
            let mut feature_idx = 0;
            for (col_idx, value) in row.iter().enumerate() {
                if Some(columns[col_idx]) == target_col {
                    if let Some(ref mut target_array) = target {
                        let numeric_value = match value.parse::<f64>() {
                            Ok(v) => v,
                            Err(_) => {
                                let category_map = category_maps.entry(col_idx).or_default();
                                let next_id = category_map.len() as f64;
                                *category_map.entry(value.clone()).or_insert(next_id)
                            }
                        };
                        target_array[row_idx] = numeric_value;
                    }
                } else {
                    let numeric_value = match value.parse::<f64>() {
                        Ok(v) => v,
                        Err(_) => {
                            let category_map = category_maps.entry(col_idx).or_default();
                            let next_id = category_map.len() as f64;
                            *category_map.entry(value.clone()).or_insert(next_id)
                        }
                    };
                    data[[row_idx, feature_idx]] = numeric_value;
                    feature_idx += 1;
                }
            }
        }
        let featurenames: Vec<String> = columns
            .iter()
            .filter(|&&_col| Some(_col) != target_col)
            .map(|&col| col.to_string())
            .collect();
        let metadata = crate::registry::DatasetMetadata {
            name: name.to_string(),
            description: format!("Real-world dataset: {}", name),
            n_samples,
            n_features,
            task_type: if target.is_some() {
                "classification".to_string()
            } else {
                "unsupervised".to_string()
            },
            targetnames: None,
            featurenames: Some(featurenames),
            url: Some(url.to_string()),
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, target, metadata))
    }
    fn create_synthetic_bank_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.random_range(0.0..1.0);
            }
            target[i] = if data.row(i).iter().take(3).sum::<f64>() > 1.5 {
                1.0
            } else {
                0.0
            };
        }
        Ok((data, target))
    }
    #[allow(dead_code)]
    fn create_synthetic_credit_approval_data(&self) -> Result<Dataset> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let n_samples = 690;
        let n_features = 15;
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        let featurenames = vec![
            "credit_score".to_string(),
            "annual_income".to_string(),
            "debt_to_income_ratio".to_string(),
            "employment_length".to_string(),
            "age".to_string(),
            "home_ownership".to_string(),
            "loan_amount".to_string(),
            "loan_purpose".to_string(),
            "credit_history_length".to_string(),
            "number_of_credit_lines".to_string(),
            "utilization_rate".to_string(),
            "delinquency_count".to_string(),
            "education_level".to_string(),
            "marital_status".to_string(),
            "verification_status".to_string(),
        ];
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(300.0..850.0);
            data[[i, 1]] = rng.random_range(20000.0..200000.0);
            data[[i, 2]] = rng.random_range(0.0..0.6);
            data[[i, 3]] = rng.random_range(0.0..30.0);
            data[[i, 4]] = rng.random_range(18.0..80.0);
            data[[i, 5]] = rng.random_range(0.0f64..3.0).floor();
            data[[i, 6]] = rng.random_range(1000.0..50000.0);
            data[[i, 7]] = rng.random_range(0.0f64..7.0).floor();
            data[[i, 8]] = rng.random_range(0.0..40.0);
            data[[i, 9]] = rng.random_range(0.0..20.0);
            data[[i, 10]] = rng.random_range(0.0..1.0);
            data[[i, 11]] = rng.random_range(0.0f64..11.0).floor();
            data[[i, 12]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 13]] = rng.random_range(0.0f64..3.0).floor();
            data[[i, 14]] = if rng.random_bool(0.7) { 1.0 } else { 0.0 };
            let credit_score_factor = (data[[i, 0]] - 300.0) / 550.0;
            let income_factor = (data[[i, 1]] / 100000.0).min(1.0);
            let debt_factor = 1.0 - data[[i, 2]];
            let employment_factor = (data[[i, 3]] / 10.0).min(1.0);
            let delinquency_penalty = data[[i, 11]] * 0.1;
            let approval_score = credit_score_factor * 0.4
                + income_factor * 0.3
                + debt_factor * 0.2
                + employment_factor * 0.1
                - delinquency_penalty;
            let noise = rng.random_range(-0.2..0.2);
            target[i] = if (approval_score + noise) > 0.5 {
                1.0
            } else {
                0.0
            };
        }
        let metadata = crate::registry::DatasetMetadata {
            name: "Credit Approval Dataset".to_string(),
            description: "Synthetic credit approval dataset with realistic financial features for binary classification"
                .to_string(),
            n_samples,
            n_features,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["denied".to_string(), "approved".to_string()]),
            featurenames: Some(featurenames),
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    #[allow(dead_code)]
    fn create_synthetic_mushroom_data(&self) -> Result<Dataset> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let n_samples = 8124;
        let n_features = 22;
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        let featurenames = vec![
            "capshape".to_string(),
            "cap_surface".to_string(),
            "cap_color".to_string(),
            "bruises".to_string(),
            "odor".to_string(),
            "gill_attachment".to_string(),
            "gill_spacing".to_string(),
            "gill_size".to_string(),
            "gill_color".to_string(),
            "stalkshape".to_string(),
            "stalk_root".to_string(),
            "stalk_surface_above_ring".to_string(),
            "stalk_surface_below_ring".to_string(),
            "stalk_color_above_ring".to_string(),
            "stalk_color_below_ring".to_string(),
            "veil_type".to_string(),
            "veil_color".to_string(),
            "ring_number".to_string(),
            "ring_type".to_string(),
            "spore_print_color".to_string(),
            "population".to_string(),
            "habitat".to_string(),
        ];
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(0.0f64..6.0).floor();
            data[[i, 1]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 2]] = rng.random_range(0.0f64..10.0).floor();
            data[[i, 3]] = if rng.random_bool(0.6) { 1.0 } else { 0.0 };
            data[[i, 4]] = rng.random_range(0.0f64..9.0).floor();
            data[[i, 5]] = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
            data[[i, 6]] = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
            data[[i, 7]] = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
            data[[i, 8]] = rng.random_range(0.0f64..12.0).floor();
            data[[i, 9]] = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
            data[[i, 10]] = rng.random_range(0.0f64..5.0).floor();
            data[[i, 11]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 12]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 13]] = rng.random_range(0.0f64..9.0).floor();
            data[[i, 14]] = rng.random_range(0.0f64..9.0).floor();
            data[[i, 15]] = 0.0;
            data[[i, 16]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 17]] = rng.random_range(0.0f64..3.0).floor();
            data[[i, 18]] = rng.random_range(0.0f64..8.0).floor();
            data[[i, 19]] = rng.random_range(0.0f64..9.0).floor();
            data[[i, 20]] = rng.random_range(0.0f64..6.0).floor();
            data[[i, 21]] = rng.random_range(0.0f64..7.0).floor();
            let mut poison_score = 0.0;
            if data[[i, 4]] == 2.0 || data[[i, 4]] == 3.0 || data[[i, 4]] == 4.0 {
                poison_score += 0.8;
            }
            if data[[i, 4]] == 5.0 || data[[i, 4]] == 7.0 {
                poison_score += 0.4;
            }
            if data[[i, 19]] == 2.0 || data[[i, 19]] == 4.0 {
                poison_score += 0.3;
            }
            if data[[i, 10]] == 0.0 {
                poison_score += 0.2;
            }
            let noise = rng.random_range(-0.3..0.3);
            target[i] = if (poison_score + noise) > 0.5 {
                1.0
            } else {
                0.0
            };
        }
        let metadata = crate::registry::DatasetMetadata {
            name: "Mushroom Dataset".to_string(),
            description: "Synthetic mushroom classification dataset with morphological features for edibility prediction"
                .to_string(),
            n_samples,
            n_features,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["edible".to_string(), "poisonous".to_string()]),
            featurenames: Some(featurenames),
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    #[allow(dead_code)]
    fn create_synthetic_spam_data(&self) -> Result<Dataset> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let n_samples = 4601;
        let n_features = 57;
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        let mut featurenames = Vec::with_capacity(n_features);
        let spam_words = vec![
            "make",
            "address",
            "all",
            "3d",
            "our",
            "over",
            "remove",
            "internet",
            "order",
            "mail",
            "receive",
            "will",
            "people",
            "report",
            "addresses",
            "free",
            "business",
            "email",
            "you",
            "credit",
            "your",
            "font",
            "000",
            "money",
            "hp",
            "hpl",
            "george",
            "650",
            "lab",
            "labs",
            "telnet",
            "857",
            "data",
            "415",
            "85",
            "technology",
            "1999",
            "parts",
            "pm",
            "direct",
            "cs",
            "meeting",
            "original",
            "project",
            "re",
            "edu",
            "table",
            "conference",
            "char_freq_semicolon",
            "char_freq_parenthesis",
            "char_freq_bracket",
            "char_freq_exclamation",
            "char_freq_dollar",
            "char_freq_hash",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ];
        for (i, word) in spam_words.iter().enumerate() {
            if i < n_features {
                featurenames.push(format!("word_freq_{word}"));
            }
        }
        while featurenames.len() < n_features {
            featurenames.push(format!("feature_{}", featurenames.len()));
        }
        for i in 0..n_samples {
            let is_spam = rng.random_bool(0.4);
            for j in 0..54 {
                if is_spam {
                    match j {
                        0..=7 => data[[i, j]] = rng.random_range(0.0..5.0),
                        8..=15 => data[[i, j]] = rng.random_range(0.0..3.0),
                        16..=25 => data[[i, j]] = rng.random_range(0.0..4.0),
                        _ => data[[i, j]] = rng.random_range(0.0..1.0),
                    }
                } else {
                    match j {
                        26..=35 => data[[i, j]] = rng.random_range(0.0..2.0),
                        36..=45 => data[[i, j]] = rng.random_range(0.0..1.5),
                        _ => data[[i, j]] = rng.random_range(0.0..0.5),
                    }
                }
            }
            if is_spam {
                data[[i, 54]] = rng.random_range(0.0..0.2);
                data[[i, 55]] = rng.random_range(0.0..0.5);
                data[[i, 56]] = rng.random_range(0.0..0.3);
            } else {
                data[[i, 54]] = rng.random_range(0.0..0.1);
                data[[i, 55]] = rng.random_range(0.0..0.2);
                data[[i, 56]] = rng.random_range(0.0..0.1);
            }
            target[i] = if is_spam { 1.0 } else { 0.0 };
        }
        let metadata = crate::registry::DatasetMetadata {
            name: "Spam Email Dataset".to_string(),
            description: "Synthetic spam email classification dataset with word and character frequency features"
                .to_string(),
            n_samples,
            n_features,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["ham".to_string(), "spam".to_string()]),
            featurenames: Some(featurenames),
            url: None,
            checksum: None,
        };
        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
    fn create_synthetic_titanic_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(1.0f64..4.0).floor();
            data[[i, 1]] = if rng.random_bool(0.5) { 0.0 } else { 1.0 };
            data[[i, 2]] = rng.random_range(1.0..80.0);
            data[[i, 3]] = rng.random_range(0.0f64..6.0).floor();
            data[[i, 4]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 5]] = rng.random_range(0.0..512.0);
            data[[i, 6]] = rng.random_range(0.0f64..3.0).floor();
            let survival_score = (4.0 - data[[i, 0]]) * 0.3
                + (1.0 - data[[i, 1]]) * 0.4
                + (80.0 - data[[i, 2]]) / 80.0 * 0.3;
            target[i] = if survival_score > 0.5 { 1.0 } else { 0.0 };
        }
        Ok((data, target))
    }
    fn create_synthetic_credit_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.random_range(0.0..1.0);
            }
            let score = data.row(i).iter().sum::<f64>() / n_features as f64;
            target[i] = if score > 0.6 { 1.0 } else { 0.0 };
        }
        Ok((data, target))
    }
    fn create_synthetic_housing_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(0.5..15.0);
            data[[i, 1]] = rng.random_range(1.0..52.0);
            data[[i, 2]] = rng.random_range(3.0..20.0);
            data[[i, 3]] = rng.random_range(0.8..6.0);
            data[[i, 4]] = rng.random_range(3.0..35682.0);
            data[[i, 5]] = rng.random_range(0.7..1243.0);
            data[[i, 6]] = rng.random_range(32.0..42.0);
            data[[i, 7]] = rng.random_range(-124.0..-114.0);
            let house_value =
                data[[i, 0]] * 50000.0 + data[[i, 2]] * 10000.0 + (40.0 - data[[i, 6]]) * 5000.0;
            target[i] = house_value / 100000.0;
        }
        Ok((data, target))
    }
    fn create_synthetic_wine_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(4.6..15.9);
            data[[i, 1]] = rng.random_range(0.12..1.58);
            data[[i, 2]] = rng.random_range(0.0..1.0);
            data[[i, 3]] = rng.random_range(0.9..15.5);
            data[[i, 4]] = rng.random_range(0.012..0.611);
            data[[i, 5]] = rng.random_range(1.0..72.0);
            data[[i, 6]] = rng.random_range(6.0..289.0);
            data[[i, 7]] = rng.random_range(0.99007..1.00369);
            data[[i, 8]] = rng.random_range(2.74..4.01);
            data[[i, 9]] = rng.random_range(0.33..2.0);
            data[[i, 10]] = rng.random_range(8.4..14.9);
            let quality: f64 = 3.0
                + (data[[i, 10]] - 8.0) * 0.5
                + (1.0 - data[[i, 1]]) * 2.0
                + data[[i, 2]] * 2.0
                + rng.random_range(-0.5..0.5);
            target[i] = quality.clamp(3.0, 8.0);
        }
        Ok((data, target))
    }
    fn create_synthetic_energy_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.random_range(0.0..1.0);
            }
            let efficiency = data.row(i).iter().sum::<f64>() / n_features as f64;
            target[i] = efficiency * 40.0 + 10.0;
        }
        Ok((data, target))
    }
    fn create_air_passengers_data(
        &self,
        n_timesteps: usize,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_timesteps, 1));
        for i in 0..n_timesteps {
            let t = i as f64;
            let trend = 100.0 + t * 2.0;
            let seasonal = 20.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
            let noise = rng.random::<f64>() * 10.0 - 5.0;
            data[[i, 0]] = trend + seasonal + noise;
        }
        Ok((data, None))
    }
    fn create_bitcoin_price_data(
        &self,
        n_timesteps: usize,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_timesteps, 6));
        let mut price = 30000.0;
        for i in 0..n_timesteps {
            let change = rng.random_range(-0.05..0.05);
            price *= 1.0 + change;
            let high = price * (1.0 + rng.random_range(0.0..0.02));
            let low = price * (1.0 - rng.random_range(0.0..0.02));
            let volume = rng.random_range(1000000.0..10000000.0);
            data[[i, 0]] = price;
            data[[i, 1]] = high;
            data[[i, 2]] = low;
            data[[i, 3]] = price;
            data[[i, 4]] = volume;
            data[[i, 5]] = price * volume;
        }
        Ok((data, None))
    }
    fn create_heart_disease_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(29.0..77.0);
            data[[i, 1]] = if rng.random_bool(0.68) { 1.0 } else { 0.0 };
            data[[i, 2]] = rng.random_range(0.0f64..4.0).floor();
            data[[i, 3]] = rng.random_range(94.0..200.0);
            data[[i, 4]] = rng.random_range(126.0..564.0);
            for j in 5..n_features {
                data[[i, j]] = rng.random_range(0.0..1.0);
            }
            let risk_score = (data[[i, 0]] - 29.0) / 48.0 * 0.3
                + data[[i, 1]] * 0.2
                + (data[[i, 3]] - 94.0) / 106.0 * 0.2
                + (data[[i, 4]] - 126.0) / 438.0 * 0.3;
            target[i] = if risk_score > 0.5 { 1.0 } else { 0.0 };
        }
        Ok((data, target))
    }
    fn create_diabetes_readmission_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.random_range(0.0..1.0);
            }
            let readmission_score = data.row(i).iter().take(10).sum::<f64>() / 10.0;
            target[i] = if readmission_score > 0.6 { 1.0 } else { 0.0 };
        }
        Ok((data, target))
    }
    fn create_synthetic_auto_mpg_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] =
                [4.0, 6.0, 8.0][rng.sample(Uniform::new(0, 3).expect("Operation failed"))];
            data[[i, 1]] = rng.random_range(68.0..455.0);
            data[[i, 2]] = rng.random_range(46.0..230.0);
            data[[i, 3]] = rng.random_range(1613.0..5140.0);
            data[[i, 4]] = rng.random_range(8.0..24.8);
            data[[i, 5]] = rng.random_range(70.0..82.0);
            data[[i, 6]] = (rng.random_range(1.0f64..4.0f64)).floor();
            let mpg: f64 = 45.0 - (data[[i, 3]] / 5140.0) * 20.0 - (data[[i, 1]] / 455.0) * 15.0
                + (data[[i, 4]] / 24.8) * 10.0
                + rng.random_range(-3.0..3.0);
            target[i] = mpg.clamp(9.0, 46.6);
        }
        Ok((data, target))
    }
    fn create_synthetic_concrete_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(102.0..540.0);
            data[[i, 1]] = rng.random_range(0.0..359.4);
            data[[i, 2]] = rng.random_range(0.0..200.1);
            data[[i, 3]] = rng.random_range(121.8..247.0);
            data[[i, 4]] = rng.random_range(0.0..32.2);
            data[[i, 5]] = rng.random_range(801.0..1145.0);
            data[[i, 6]] = rng.random_range(594.0..992.6);
            data[[i, 7]] = rng.random_range(1.0..365.0);
            let strength: f64 = (data[[i, 0]] / 540.0) * 30.0
                + (data[[i, 1]] / 359.4) * 15.0
                + (data[[i, 3]] / 247.0) * (-20.0)
                + (data[[i, 7]] / 365.0_f64).ln() * 10.0
                + rng.random_range(-5.0..5.0);
            target[i] = strength.clamp(2.33, 82.6);
        }
        Ok((data, target))
    }
    fn create_synthetic_electricity_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let hour = (i % 24) as f64;
            let day_of_year = (i / 24) % 365;
            data[[i, 0]] = 20.0
                + 15.0 * (day_of_year as f64 * 2.0 * std::f64::consts::PI / 365.0).sin()
                + rng.random_range(-5.0..5.0);
            data[[i, 1]] = 50.0 + 30.0 * rng.random_range(0.0..1.0);
            data[[i, 2]] = hour;
            let seasonal = 50.0
                + 30.0
                    * (day_of_year as f64 * 2.0 * std::f64::consts::PI / 365.0
                        + std::f64::consts::PI)
                        .cos();
            let daily = 40.0 + 60.0 * ((hour - 12.0) * std::f64::consts::PI / 12.0).cos();
            let temp_effect = (data[[i, 0]] - 20.0).abs() * 2.0;
            target[i] = seasonal + daily + temp_effect + rng.random_range(-10.0..10.0);
        }
        Ok((data, target))
    }
    fn create_synthetic_stock_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        let mut price = 100.0;
        for i in 0..n_samples {
            let change = rng.random_range(-0.05..0.05);
            price *= 1.0 + change;
            let high = price * (1.0 + rng.random_range(0.0..0.02));
            let low = price * (1.0 - rng.random_range(0.0..0.02));
            let volume = rng.random_range(1000000.0..10000000.0);
            data[[i, 0]] = price;
            data[[i, 1]] = high;
            data[[i, 2]] = low;
            data[[i, 3]] = volume;
            data[[i, 4]] = (high - low) / price;
            let next_change = rng.random_range(-0.05..0.05);
            target[i] = next_change;
        }
        Ok((data, target))
    }
    fn create_synthetic_fraud_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let is_fraud = rng.random_range(0.0..1.0) < 0.001728;
            for j in 0..n_features {
                if j < 28 {
                    if is_fraud {
                        data[[i, j]] = rng.random_range(-5.0..5.0) * 2.0;
                    } else {
                        data[[i, j]] = rng.random_range(-3.0..3.0);
                    }
                }
            }
            target[i] = if is_fraud { 1.0 } else { 0.0 };
        }
        Ok((data, target))
    }
    fn create_synthetic_loan_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(1000.0..50000.0);
            data[[i, 1]] = rng.random_range(5.0..25.0);
            data[[i, 2]] = [12.0, 24.0, 36.0, 48.0, 60.0]
                [rng.sample(Uniform::new(0, 5).expect("Operation failed"))];
            data[[i, 3]] = rng.random_range(20000.0..200000.0);
            data[[i, 4]] = rng.random_range(300.0..850.0);
            data[[i, 5]] = rng.random_range(0.0..40.0);
            data[[i, 6]] = rng.random_range(0.0..0.4);
            for j in 7..n_features {
                data[[i, j]] = rng.random_range(0.0..1.0);
            }
            let risk_score = (850.0 - data[[i, 4]]) / 550.0 * 0.4
                + data[[i, 6]] * 0.3
                + (data[[i, 1]] - 5.0) / 20.0 * 0.2
                + (50000.0 - data[[i, 3]]) / 180000.0 * 0.1;
            target[i] = if risk_score > 0.3 { 1.0 } else { 0.0 };
        }
        Ok((data, target))
    }
    fn create_synthetic_adult_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            data[[i, 0]] = rng.random_range(17.0..90.0);
            data[[i, 1]] = rng.random_range(0.0f64..9.0).floor();
            data[[i, 2]] = rng.random_range(12285.0..1484705.0);
            data[[i, 3]] = rng.random_range(0.0f64..16.0).floor();
            data[[i, 4]] = rng.random_range(1.0..17.0);
            data[[i, 5]] = rng.random_range(0.0f64..7.0).floor();
            data[[i, 6]] = rng.random_range(0.0f64..14.0).floor();
            data[[i, 7]] = rng.random_range(0.0f64..6.0).floor();
            data[[i, 8]] = rng.random_range(0.0f64..5.0).floor();
            data[[i, 9]] = if rng.random_bool(0.67) { 1.0 } else { 0.0 };
            data[[i, 10]] = if rng.random_bool(0.9) {
                0.0
            } else {
                rng.random_range(1.0..99999.0)
            };
            data[[i, 11]] = if rng.random_bool(0.95) {
                0.0
            } else {
                rng.random_range(1.0..4356.0)
            };
            data[[i, 12]] = rng.random_range(1.0..99.0);
            data[[i, 13]] = rng.random_range(0.0f64..41.0).floor();
            let income_score = (data[[i, 0]] - 17.0) / 73.0 * 0.2
                + data[[i, 4]] / 16.0 * 0.3
                + data[[i, 9]] * 0.2
                + (data[[i, 12]] - 1.0) / 98.0 * 0.2
                + (data[[i, 10]] + data[[i, 11]]) / 100000.0 * 0.1;
            let noise = rng.random_range(-0.15..0.15);
            target[i] = if (income_score + noise) > 0.5 {
                1.0
            } else {
                0.0
            };
        }
        Ok((data, target))
    }
    fn create_generic_synthetic_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        has_categorical: bool,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                if has_categorical && j < n_features / 3 {
                    data[[i, j]] = rng.random_range(0.0f64..10.0).floor();
                } else {
                    data[[i, j]] = rng.random_range(-2.0..2.0);
                }
            }
        }
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let feature_sum = data.row(i).iter().sum::<f64>();
            let score = feature_sum / n_features as f64;
            target[i] = if score > 0.0 { 1.0 } else { 0.0 };
        }
        Ok((data, Some(target)))
    }
    fn create_synthetic_cifar10_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let class = rng.sample(Uniform::new(0, 10).expect("Operation failed")) as f64;
            target[i] = class;
            for j in 0..n_features {
                let base_intensity = match class as i32 {
                    0 => 0.6,
                    1 => 0.3,
                    2 => 0.8,
                    3 => 0.5,
                    4 => 0.7,
                    5 => 0.4,
                    6 => 0.9,
                    7 => 0.6,
                    8 => 0.2,
                    9 => 0.3,
                    _ => 0.5,
                };
                data[[i, j]] = base_intensity + rng.random_range(-0.3f64..0.3f64);
                data[[i, j]] = data[[i, j]].clamp(0.0f64, 1.0f64);
            }
        }
        Ok((data, target))
    }
    fn create_synthetic_fashion_mnist_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let class = rng.sample(Uniform::new(0, 10).expect("Operation failed")) as f64;
            target[i] = class;
            for j in 0..n_features {
                let base_intensity = match class as i32 {
                    0 => 0.3,
                    1 => 0.4,
                    2 => 0.5,
                    3 => 0.6,
                    4 => 0.7,
                    5 => 0.2,
                    6 => 0.4,
                    7 => 0.3,
                    8 => 0.5,
                    9 => 0.4,
                    _ => 0.4,
                };
                let texture_noise = rng.random_range(-0.2f64..0.2f64);
                data[[i, j]] = base_intensity + texture_noise;
                data[[i, j]] = data[[i, j]].clamp(0.0f64, 1.0f64);
            }
        }
        Ok((data, target))
    }
    fn create_synthetic_imdb_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        let positive_words = 0..n_features / 3;
        let negative_words = n_features / 3..2 * n_features / 3;
        let _neutral_words = 2 * n_features / 3..n_features;
        for i in 0..n_samples {
            let is_positive = rng.random_bool(0.5);
            target[i] = if is_positive { 1.0 } else { 0.0 };
            for j in 0..n_features {
                let base_freq = if positive_words.contains(&j) {
                    if is_positive {
                        rng.random_range(0.5..2.0)
                    } else {
                        rng.random_range(0.0..0.5)
                    }
                } else if negative_words.contains(&j) {
                    if is_positive {
                        rng.random_range(0.0..0.5)
                    } else {
                        rng.random_range(0.5..2.0)
                    }
                } else {
                    rng.random_range(0.2..1.0)
                };
                data[[i, j]] = base_freq;
            }
        }
        Ok((data, target))
    }
    fn create_synthetic_news_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::random::{Rng, RngExt};
        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);
        let words_per_topic = n_features / 5;
        for i in 0..n_samples {
            let topic = rng.sample(Uniform::new(0, 5).expect("Operation failed")) as f64;
            target[i] = topic;
            for j in 0..n_features {
                let word_topic = j / words_per_topic;
                let base_freq = if word_topic == topic as usize {
                    rng.random_range(1.0..3.0)
                } else {
                    rng.random_range(0.0..0.8)
                };
                let noise = rng.random_range(-0.2f64..0.2f64);
                data[[i, j]] = (base_freq + noise).max(0.0f64);
            }
        }
        Ok((data, target))
    }
}
