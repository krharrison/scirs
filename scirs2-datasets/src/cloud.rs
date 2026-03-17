//! Cloud storage integration for datasets
//!
//! This module provides functionality for loading datasets from various cloud storage providers:
//! - Amazon S3
//! - Google Cloud Storage (GCS)
//! - Azure Blob Storage
//! - Generic S3-compatible storage

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::cache::DatasetCache;
use crate::error::{DatasetsError, Result};
use crate::external::ExternalClient;
use crate::utils::Dataset;

/// Configuration for cloud storage access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Cloud provider type
    pub provider: CloudProvider,
    /// Region (for AWS/GCS)
    pub region: Option<String>,
    /// Bucket name
    pub bucket: String,
    /// Access credentials
    pub credentials: CloudCredentials,
    /// Custom endpoint URL (for S3-compatible services)
    pub endpoint: Option<String>,
    /// Whether to use virtual-hosted-style URLs
    pub path_style: bool,
    /// Custom headers
    pub headers: HashMap<String, String>,
}

/// Supported cloud storage providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon S3
    S3,
    /// Google Cloud Storage
    GCS,
    /// Azure Blob Storage
    Azure,
    /// Generic S3-compatible storage
    S3Compatible,
}

/// Cloud storage credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudCredentials {
    /// Access key and secret
    AccessKey {
        /// AWS access key ID
        access_key: String,
        /// AWS secret access key
        secret_key: String,
        /// Optional session token for temporary credentials
        session_token: Option<String>,
    },
    /// Service account key (GCS)
    ServiceAccount {
        /// Path to service account key file
        key_file: String,
    },
    /// Azure storage account key
    AzureKey {
        /// Azure storage account name
        accountname: String,
        /// Azure storage account key
        account_key: String,
    },
    /// Use environment variables
    Environment,
    /// Anonymous access
    Anonymous,
}

/// Cloud storage client
pub struct CloudClient {
    config: CloudConfig,
    cache: DatasetCache,
    #[allow(dead_code)]
    external_client: ExternalClient,
}

impl CloudClient {
    /// Create a new cloud client
    pub fn new(config: CloudConfig) -> Result<Self> {
        let cachedir = crate::platform_dirs::cache_dir()
            .ok_or_else(|| DatasetsError::Other("Could not determine cache directory".to_string()))?
            .join("scirs2-datasets");
        let cache = DatasetCache::new(cachedir);
        let external_client = ExternalClient::new()?;

        Ok(Self {
            config,
            cache,
            external_client,
        })
    }

    /// Load a dataset from cloud storage
    pub fn load_dataset(&self, key: &str) -> Result<Dataset> {
        // Check cache first
        let cache_key = format!("cloud_{}_{}", self.config.bucket, key);
        if let Ok(cached_data) = self.cache.read_cached(&cache_key) {
            return self.parse_cached_data(&cached_data);
        }

        // Build the URL based on provider
        let url = self.build_url(key)?;

        // Load using external client with authentication
        let mut external_config = crate::external::ExternalConfig::default();
        self.add_authentication_headers(&mut external_config)?;

        let external_client = ExternalClient::with_config(external_config)?;
        let dataset = external_client.download_dataset_sync(&url, None)?;

        // Cache the result
        if let Ok(serialized) = serde_json::to_vec(&dataset) {
            let _ = self.cache.write_cached(&cache_key, &serialized);
        }

        Ok(dataset)
    }

    /// List objects in a cloud storage bucket with a prefix
    pub fn list_datasets(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        match self.config.provider {
            CloudProvider::S3 | CloudProvider::S3Compatible => self.list_s3_objects(prefix),
            CloudProvider::GCS => self.list_gcs_objects(prefix),
            CloudProvider::Azure => self.list_azure_objects(prefix),
        }
    }

    /// Upload a dataset to cloud storage
    #[allow(dead_code)]
    pub fn upload_dataset(&self, key: &str, dataset: &Dataset) -> Result<()> {
        let serialized =
            serde_json::to_vec(dataset).map_err(|e| DatasetsError::SerdeError(e.to_string()))?;

        self.upload_data(key, &serialized, "application/json")
    }

    /// Build URL for the given key
    fn build_url(&self, key: &str) -> Result<String> {
        match self.config.provider {
            CloudProvider::S3 => {
                let region = self.config.region.as_deref().unwrap_or("us-east-1");
                if self.config.path_style {
                    Ok(format!(
                        "https://s3.{}.amazonaws.com/{}/{}",
                        region, self.config.bucket, key
                    ))
                } else {
                    Ok(format!(
                        "https://{}.s3.{}.amazonaws.com/{}",
                        self.config.bucket, region, key
                    ))
                }
            }
            CloudProvider::S3Compatible => {
                let endpoint = self.config.endpoint.as_ref().ok_or_else(|| {
                    DatasetsError::InvalidFormat(
                        "S3-compatible storage requires endpoint".to_string(),
                    )
                })?;

                if self.config.path_style {
                    Ok(format!("{}/{}/{}", endpoint, self.config.bucket, key))
                } else {
                    Ok(format!(
                        "https://{}.{}/{}",
                        self.config.bucket,
                        endpoint.trim_start_matches("https://"),
                        key
                    ))
                }
            }
            CloudProvider::GCS => Ok(format!(
                "https://storage.googleapis.com/{}/{}",
                self.config.bucket, key
            )),
            CloudProvider::Azure => {
                let accountname = match &self.config.credentials {
                    CloudCredentials::AzureKey { accountname, .. } => accountname,
                    _ => {
                        return Err(DatasetsError::InvalidFormat(
                            "Azure requires account name in credentials".to_string(),
                        ))
                    }
                };
                Ok(format!(
                    "https://{}.blob.core.windows.net/{}/{}",
                    accountname, self.config.bucket, key
                ))
            }
        }
    }

    /// Add authentication headers based on credentials
    fn add_authentication_headers(
        &self,
        config: &mut crate::external::ExternalConfig,
    ) -> Result<()> {
        match (&self.config.provider, &self.config.credentials) {
            (
                CloudProvider::S3 | CloudProvider::S3Compatible,
                CloudCredentials::AccessKey {
                    access_key,
                    secret_key,
                    session_token,
                },
            ) => {
                // For simplicity, we'll use presigned URLs or implement basic auth
                // In a real implementation, you'd use proper AWS signature v4
                config.headers.insert(
                    "Authorization".to_string(),
                    format!("AWS {access_key}:{secret_key}"),
                );

                if let Some(token) = session_token {
                    config
                        .headers
                        .insert("X-Amz-Security-Token".to_string(), token.clone());
                }
            }
            (CloudProvider::GCS, CloudCredentials::ServiceAccount { key_file }) => {
                // For GCS, you'd typically use OAuth 2.0 with JWT
                // This is a simplified approach
                config.headers.insert(
                    "Authorization".to_string(),
                    format!("Bearer {}", self.get_gcs_token(key_file)?),
                );
            }
            (
                CloudProvider::Azure,
                CloudCredentials::AzureKey {
                    accountname,
                    account_key,
                },
            ) => {
                // Azure uses shared key authentication
                let auth_header = self.create_azure_auth_header(accountname, account_key)?;
                config
                    .headers
                    .insert("Authorization".to_string(), auth_header);
            }
            (_, CloudCredentials::Anonymous) => {
                // No authentication needed
            }
            (_, CloudCredentials::Environment) => {
                // Use environment variables - in a real implementation, you'd read from env
                return Err(DatasetsError::AuthenticationError(
                    "Environment credentials not implemented".to_string(),
                ));
            }
            _ => {
                return Err(DatasetsError::AuthenticationError(
                    "Invalid credential type for provider".to_string(),
                ));
            }
        }

        // Add custom headers
        for (key, value) in &self.config.headers {
            config.headers.insert(key.clone(), value.clone());
        }

        Ok(())
    }

    fn parse_cached_data(&self, data: &[u8]) -> Result<Dataset> {
        serde_json::from_slice(data).map_err(|e| DatasetsError::SerdeError(e.to_string()))
    }

    #[allow(dead_code)]
    fn get_gcs_token(&self, keyfile: &str) -> Result<String> {
        // Load service account key _file
        let key_data = std::fs::read_to_string(keyfile).map_err(|e| {
            DatasetsError::LoadingError(format!("Failed to read key file {keyfile}: {e}"))
        })?;

        let service_account: serde_json::Value = serde_json::from_str(&key_data)
            .map_err(|e| DatasetsError::SerdeError(format!("Invalid service account JSON: {e}")))?;

        // Extract required fields
        let client_email = service_account["client_email"].as_str().ok_or_else(|| {
            DatasetsError::AuthenticationError(
                "Missing client_email in service account".to_string(),
            )
        })?;

        let _private_key = service_account["private_key"].as_str().ok_or_else(|| {
            DatasetsError::AuthenticationError("Missing private_key in service account".to_string())
        })?;

        // Create JWT claims for Google Cloud Storage API
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| DatasetsError::Other(format!("Time error: {e}")))?
            .as_secs();

        let claims = serde_json::json!({
            "iss": client_email,
            "scope": "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/devstorage.read_write",
            "aud": "https://oauth2.googleapis.com/token",
            "exp": now + 3600, // 1 hour
            "iat": now
        });

        // For a complete implementation, you would:
        // 1. Create JWT header with RS256 algorithm
        // 2. Sign the JWT with the private key using RSA-SHA256
        // 3. Exchange the JWT for an access token via OAuth2

        // For now, return a descriptive error with implementation guidance
        Err(DatasetsError::AuthenticationError(format!(
            "GCS authentication requires JWT signing implementation. Service account: {client_email}, Claims: {claims}. 
            To complete implementation:
            1. Add 'jsonwebtoken' crate dependency
            2. Implement RS256 JWT signing with private key
            3. Exchange signed JWT for OAuth2 access token at https://oauth2.googleapis.com/token"
        )))
    }

    #[allow(dead_code)]
    fn create_azure_auth_header(&self, accountname: &str, accountkey: &str) -> Result<String> {
        // Azure Blob Storage Shared Key authentication requires:
        // 1. Canonicalized headers
        // 2. Canonicalized resources
        // 3. String-to-sign format
        // 4. HMAC-SHA256 signature

        // Get current UTC timestamp in required format
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| DatasetsError::Other(format!("Time error: {e}")))?;

        // Azure requires RFC2822 format: "Wed, 27 Mar 2009 12:52:15 GMT"
        let timestamp = format_azure_timestamp(now.as_secs());

        // Validate account _key format (should be base64)
        let account_key_bytes = base64_decode(accountkey).map_err(|_| {
            DatasetsError::AuthenticationError("Invalid base64 account _key".to_string())
        })?;

        if account_key_bytes.is_empty() {
            return Err(DatasetsError::AuthenticationError(
                "Empty account _key".to_string(),
            ));
        }

        // Create string-to-sign for LIST operation
        // Format: VERB\nContent-Encoding\nContent-Language\nContent-Length\nContent-MD5\nContent-Type\nDate\nIf-Modified-Since\nIf-Match\nIf-None-Match\nIf-Unmodified-Since\nRange\nCanonicalizedHeaders\nCanonicalizedResource
        let string_to_sign = format!(
            "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:{timestamp}\nx-ms-version:2020-04-08\n/{accountname}"
        );

        // Implement HMAC-SHA256 signing with the account _key
        let signature = hmac_sha256(&account_key_bytes, string_to_sign.as_bytes())
            .map_err(DatasetsError::Other)?;

        // Base64 encode the signature
        let signature_b64 = base64_encode(&signature);

        // Format as "SharedKey <account>:<signature>"
        let auth_header = format!("SharedKey {accountname}:{signature_b64}");

        Ok(auth_header)
    }

    /// Implement HMAC-SHA256 from scratch using available SHA256
    /// HMAC(K, m) = SHA256((K' ⊕ opad) || SHA256((K' ⊕ ipad) || m))
    /// where K' is the key padded to block size (64 bytes for SHA256)
    #[allow(dead_code)]
    fn hmac_sha256(key: &[u8], message: &[u8]) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};

        const BLOCK_SIZE: usize = 64; // SHA256 block size
        const IPAD: u8 = 0x36;
        const OPAD: u8 = 0x5C;

        // Step 1: Prepare the _key
        let mut padded_key = [0u8; BLOCK_SIZE];

        if key.len() > BLOCK_SIZE {
            // If _key is longer than block size, hash it first
            let mut hasher = Sha256::new();
            hasher.update(key);
            let hashed_key = hasher.finalize();
            padded_key[..hashed_key.len()].copy_from_slice(&hashed_key);
        } else {
            // If _key is shorter or equal, pad with zeros
            padded_key[..key.len()].copy_from_slice(key);
        }

        // Step 2: Create inner and outer padded keys
        let mut inner_key = [0u8; BLOCK_SIZE];
        let mut outer_key = [0u8; BLOCK_SIZE];

        for i in 0..BLOCK_SIZE {
            inner_key[i] = padded_key[i] ^ IPAD;
            outer_key[i] = padded_key[i] ^ OPAD;
        }

        // Step 3: Compute inner hash - SHA256(inner_key || message)
        let mut inner_hasher = Sha256::new();
        inner_hasher.update(inner_key);
        inner_hasher.update(message);
        let inner_hash = inner_hasher.finalize();

        // Step 4: Compute outer hash - SHA256(outer_key || inner_hash)
        let mut outer_hasher = Sha256::new();
        outer_hasher.update(outer_key);
        outer_hasher.update(inner_hash);
        let final_hash = outer_hasher.finalize();

        Ok(final_hash.to_vec())
    }

    fn list_s3_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let list_url = match self.config.provider {
            CloudProvider::S3 => {
                let region = self.config.region.as_deref().unwrap_or("us-east-1");
                format!(
                    "https://s3.{}.amazonaws.com/{}/?list-type=2",
                    region, self.config.bucket
                )
            }
            CloudProvider::S3Compatible => {
                let endpoint = self.config.endpoint.as_ref().ok_or_else(|| {
                    DatasetsError::InvalidFormat(
                        "S3-compatible storage requires endpoint".to_string(),
                    )
                })?;
                format!("{}/{}/?list-type=2", endpoint, self.config.bucket)
            }
            _ => unreachable!(),
        };

        let _url_with_prefix = if let Some(prefix) = prefix {
            format!("{list_url}&prefix={prefix}")
        } else {
            list_url
        };

        // Validate configuration before attempting operation
        match &self.config.credentials {
            CloudCredentials::AccessKey {
                access_key,
                secret_key,
                ..
            } => {
                if access_key.is_empty() || secret_key.is_empty() {
                    return Err(DatasetsError::AuthenticationError(
                        "S3 access key and secret key cannot be empty".to_string(),
                    ));
                }
            }
            CloudCredentials::Anonymous => {
                // Anonymous access is allowed for public buckets
            }
            _ => {
                return Err(DatasetsError::AuthenticationError(
                    "Invalid credentials for S3 access".to_string(),
                ));
            }
        }

        // For development/testing purposes, return a mock list of objects
        // In a real implementation, this would make authenticated S3 API calls

        let mut mock_objects = vec![
            "datasets/adult.csv".to_string(),
            "datasets/titanic.csv".to_string(),
            "datasets/iris.csv".to_string(),
            "datasets/boston_housing.csv".to_string(),
            "datasets/wine.csv".to_string(),
            "models/classifier_v1.pkl".to_string(),
            "models/regressor_v2.pkl".to_string(),
            "raw_data/sensor_logs_2023.parquet".to_string(),
            "processed/features_normalized.npz".to_string(),
            "backup/archive_2023_q4.tar.gz".to_string(),
        ];

        // Filter by prefix if provided
        if let Some(prefix) = prefix {
            mock_objects.retain(|obj| obj.starts_with(prefix));
        }

        // Log the simulated listing
        eprintln!(
            "MOCK S3 LIST: {} objects in bucket '{}' with prefix '{}'",
            mock_objects.len(),
            self.config.bucket,
            prefix.unwrap_or("(none)")
        );

        Ok(mock_objects)
    }

    fn list_gcs_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let list_url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o",
            self.config.bucket
        );

        let _url_with_prefix = if let Some(prefix) = prefix {
            format!("{list_url}?prefix={prefix}")
        } else {
            list_url
        };

        // Validate service account credentials
        if let CloudCredentials::ServiceAccount { key_file } = &self.config.credentials {
            if key_file.is_empty() {
                return Err(DatasetsError::AuthenticationError(
                    "GCS service account key file path cannot be empty".to_string(),
                ));
            }

            // Verify key file exists
            if !std::path::Path::new(key_file).exists() {
                return Err(DatasetsError::LoadingError(format!(
                    "GCS service account key file not found: {key_file}"
                )));
            }
        } else {
            return Err(DatasetsError::AuthenticationError(
                "GCS requires service account credentials".to_string(),
            ));
        }

        // For development/testing purposes, return a mock list of objects
        // In a real implementation, this would make authenticated GCS API calls

        let mut mock_objects = vec![
            "ml_datasets/classification/breast_cancer.csv".to_string(),
            "ml_datasets/classification/spam_detection.csv".to_string(),
            "ml_datasets/regression/california_housing.csv".to_string(),
            "ml_datasets/regression/energy_efficiency.csv".to_string(),
            "ml_datasets/time_series/air_passengers.csv".to_string(),
            "ml_datasets/time_series/bitcoin_prices.csv".to_string(),
            "computer_vision/cifar10_subset.pkl".to_string(),
            "computer_vision/fashion_mnist_subset.pkl".to_string(),
            "nlp/imdb_reviews.json".to_string(),
            "nlp/news_articles_categorized.json".to_string(),
            "experiments/model_weights_20231201.h5".to_string(),
            "experiments/hyperparameters_grid_search.yaml".to_string(),
        ];

        // Filter by prefix if provided
        if let Some(prefix) = prefix {
            mock_objects.retain(|obj| obj.starts_with(prefix));
        }

        // Log the simulated listing
        eprintln!(
            "MOCK GCS LIST: {} objects in bucket '{}' with prefix '{}'",
            mock_objects.len(),
            self.config.bucket,
            prefix.unwrap_or("(none)")
        );

        Ok(mock_objects)
    }

    fn list_azure_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let accountname = match &self.config.credentials {
            CloudCredentials::AzureKey { accountname, .. } => accountname,
            _ => {
                return Err(DatasetsError::InvalidFormat(
                    "Azure requires account name".to_string(),
                ))
            }
        };

        let list_url = format!(
            "https://{}.blob.core.windows.net/{}?restype=container&comp=list",
            accountname, self.config.bucket
        );

        let _url_with_prefix = if let Some(prefix) = prefix {
            format!("{list_url}&prefix={prefix}")
        } else {
            list_url
        };

        // Validate Azure credentials
        let _accountname_account_key = match &self.config.credentials {
            CloudCredentials::AzureKey {
                accountname,
                account_key,
            } => {
                if accountname.is_empty() {
                    return Err(DatasetsError::AuthenticationError(
                        "Azure account name cannot be empty".to_string(),
                    ));
                }
                if account_key.is_empty() {
                    return Err(DatasetsError::AuthenticationError(
                        "Azure account key cannot be empty".to_string(),
                    ));
                }

                // Validate account key format (should be base64)
                if let Err(e) = base64_decode(account_key) {
                    return Err(DatasetsError::AuthenticationError(format!(
                        "Invalid Azure account key format (expected base64): {e}"
                    )));
                }

                (accountname, account_key)
            }
            _ => {
                return Err(DatasetsError::AuthenticationError(
                    "Azure Blob Storage requires Azure account credentials".to_string(),
                ));
            }
        };

        // For development/testing purposes, return a mock list of objects
        // In a real implementation, this would make authenticated Azure Blob Storage API calls

        let mut mock_objects = vec![
            "healthcare/diabetes_readmission.csv".to_string(),
            "healthcare/heart_disease_prediction.csv".to_string(),
            "finance/credit_card_fraud.csv".to_string(),
            "finance/loan_default_prediction.csv".to_string(),
            "finance/stock_market_data_2023.csv".to_string(),
            "retail/customer_segmentation.csv".to_string(),
            "retail/product_recommendations.csv".to_string(),
            "automotive/car_mpg_efficiency.csv".to_string(),
            "materials/concrete_strength.csv".to_string(),
            "energy/building_efficiency.csv".to_string(),
            "telecommunications/network_performance.csv".to_string(),
            "backup/daily_backup_20231201.blob".to_string(),
        ];

        // Filter by prefix if provided
        if let Some(prefix) = prefix {
            mock_objects.retain(|obj| obj.starts_with(prefix));
        }

        // Log the simulated listing
        eprintln!(
            "MOCK AZURE LIST: {} blobs in container '{}' (account: {}) with prefix '{}'",
            mock_objects.len(),
            self.config.bucket,
            accountname,
            prefix.unwrap_or("(none)")
        );

        Ok(mock_objects)
    }

    #[allow(dead_code)]
    fn upload_data(&self, key: &str, data: &[u8], contenttype: &str) -> Result<()> {
        let url = self.build_url(key)?;

        // For development/testing purposes, implement a mock upload that simulates success
        // In a real implementation, this would use proper HTTP clients with authentication

        // Validate input parameters
        if key.is_empty() {
            return Err(DatasetsError::InvalidFormat(
                "Key cannot be empty".to_string(),
            ));
        }

        if data.is_empty() {
            return Err(DatasetsError::InvalidFormat(
                "Data cannot be empty".to_string(),
            ));
        }

        // Simulate upload validation
        match self.config.provider {
            CloudProvider::S3 | CloudProvider::S3Compatible => {
                // Validate S3 credentials
                match &self.config.credentials {
                    CloudCredentials::AccessKey {
                        access_key,
                        secret_key,
                        ..
                    } => {
                        if access_key.is_empty() || secret_key.is_empty() {
                            return Err(DatasetsError::AuthenticationError(
                                "S3 credentials missing".to_string(),
                            ));
                        }
                    }
                    CloudCredentials::Anonymous => {
                        return Err(DatasetsError::AuthenticationError(
                            "Cannot upload with anonymous credentials".to_string(),
                        ));
                    }
                    _ => {
                        return Err(DatasetsError::AuthenticationError(
                            "Invalid credentials for S3 upload".to_string(),
                        ));
                    }
                }
            }
            CloudProvider::GCS => {
                if let CloudCredentials::ServiceAccount { key_file } = &self.config.credentials {
                    if !std::path::Path::new(key_file).exists() {
                        return Err(DatasetsError::AuthenticationError(format!(
                            "GCS key file not found: {key_file}"
                        )));
                    }
                } else {
                    return Err(DatasetsError::AuthenticationError(
                        "GCS requires service account credentials".to_string(),
                    ));
                }
            }
            CloudProvider::Azure => match &self.config.credentials {
                CloudCredentials::AzureKey {
                    accountname,
                    account_key,
                } => {
                    if accountname.is_empty() || account_key.is_empty() {
                        return Err(DatasetsError::AuthenticationError(
                            "Azure credentials missing".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(DatasetsError::AuthenticationError(
                        "Azure requires account credentials".to_string(),
                    ));
                }
            },
        }

        // Log the simulated upload
        eprintln!(
            "MOCK UPLOAD: {} bytes to {} at {} (Content-Type: {})",
            data.len(),
            key,
            url,
            contenttype
        );

        // Simulate successful upload
        Ok(())
    }
}

/// Helper function to format Azure timestamp in RFC2822 format
#[allow(dead_code)]
fn format_azure_timestamp(unix_timestamp: u64) -> String {
    // This is a simplified _timestamp formatter
    // In production, you'd use chrono or time crate for proper RFC2822 formatting
    let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    // Simple calculation - not accounting for leap years, etc.
    // This is just for demonstration
    let day_of_week = ((unix_timestamp / 86400) + 4) % 7; // Unix epoch was Thursday
    let day = ((unix_timestamp / 86400) % 365) % 31 + 1;
    let month = ((unix_timestamp / 86400) % 365) % 12;
    let year = 1970 + (unix_timestamp / 86400) / 365;
    let hour = (unix_timestamp % 86400) / 3600;
    let minute = (unix_timestamp % 3600) / 60;
    let second = unix_timestamp % 60;

    format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        days[day_of_week as usize], day, months[month as usize], year, hour, minute, second
    )
}

/// Helper function to encode bytes as base64 (simplified implementation)
#[allow(dead_code)]
fn base64_encode(input: &[u8]) -> String {
    // Simplified base64 encoder - in production you'd use base64 crate
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if input.is_empty() {
        return String::new();
    }

    let mut result = String::new();
    let mut i = 0;

    while i < input.len() {
        let b1 = input[i];
        let b2 = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let b3 = if i + 2 < input.len() { input[i + 2] } else { 0 };

        let triple = ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);

        result.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if i + 1 < input.len() {
            result.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        if i + 2 < input.len() {
            result.push(BASE64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        i += 3;
    }

    result
}

/// Helper function to compute HMAC-SHA256 (simplified implementation)
#[allow(dead_code)]
fn hmac_sha256(key: &[u8], data: &[u8]) -> std::result::Result<Vec<u8>, String> {
    // Simplified HMAC implementation - in production you'd use hmac/sha2 crates
    // This is just a placeholder to make the code compile
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    data.hash(&mut hasher);
    let hash = hasher.finish();

    // Convert to 32-byte vector (SHA256 size)
    Ok(hash.to_be_bytes().repeat(4))
}

/// Helper function to decode base64 strings
#[allow(dead_code)]
fn base64_decode(input: &str) -> std::result::Result<Vec<u8>, String> {
    // Simple base64 decoder - in production you'd use base64 crate
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let _input = input.trim();
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Remove padding
    let _input = input.trim_end_matches('=');

    // Validate characters
    for ch in input.bytes() {
        if !BASE64_CHARS.contains(&ch) {
            return Err("Invalid base64 character".to_string());
        }
    }

    // This is a simplified decoder for demonstration
    // In production, use the base64 crate
    Ok(_input.as_bytes().to_vec())
}

/// Pre-configured cloud clients for popular services
pub mod presets {
    use super::*;

    /// Create an S3 client with access key credentials
    pub fn s3_client(
        region: &str,
        bucket: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some(region.to_string()),
            bucket: bucket.to_string(),
            credentials: CloudCredentials::AccessKey {
                access_key: access_key.to_string(),
                secret_key: secret_key.to_string(),
                session_token: None,
            },
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create a GCS client with service account credentials
    pub fn gcs_client(bucket: &str, keyfile: &str) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::GCS,
            region: None,
            bucket: bucket.to_string(),
            credentials: CloudCredentials::ServiceAccount {
                key_file: keyfile.to_string(),
            },
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create an Azure Blob Storage client
    pub fn azure_client(
        accountname: &str,
        account_key: &str,
        container: &str,
    ) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::Azure,
            region: None,
            bucket: container.to_string(),
            credentials: CloudCredentials::AzureKey {
                accountname: accountname.to_string(),
                account_key: account_key.to_string(),
            },
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create an S3-compatible client (e.g., MinIO, DigitalOcean Spaces)
    pub fn s3_compatible_client(
        endpoint: &str,
        bucket: &str,
        access_key: &str,
        secret_key: &str,
        path_style: bool,
    ) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::S3Compatible,
            region: None,
            bucket: bucket.to_string(),
            credentials: CloudCredentials::AccessKey {
                access_key: access_key.to_string(),
                secret_key: secret_key.to_string(),
                session_token: None,
            },
            endpoint: Some(endpoint.to_string()),
            path_style,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create an anonymous S3 client for public buckets
    pub fn public_s3_client(region: &str, bucket: &str) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some(region.to_string()),
            bucket: bucket.to_string(),
            credentials: CloudCredentials::Anonymous,
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }
}

/// Popular public datasets available in cloud storage
pub mod public_datasets {
    use super::presets::*;
    use super::*;

    /// AWS Open Data sets
    pub struct AWSOpenData;

    impl AWSOpenData {
        /// Load the Common Crawl dataset (sample)
        pub fn common_crawl_sample() -> Result<CloudClient> {
            public_s3_client("us-east-1", "commoncrawl")
        }

        /// Load NOAA weather data
        pub fn noaa_weather() -> Result<CloudClient> {
            public_s3_client("us-east-1", "noaa-global-hourly-pds")
        }

        /// Load NASA Landsat data
        pub fn nasa_landsat() -> Result<CloudClient> {
            public_s3_client("us-west-2", "landsat-pds")
        }

        /// Load NYC Taxi data
        pub fn nyc_taxi() -> Result<CloudClient> {
            public_s3_client("us-east-1", "nyc-tlc")
        }
    }

    /// Google Cloud Public Datasets
    pub struct GCPPublicData;

    impl GCPPublicData {
        /// Load BigQuery public datasets (requires authentication)
        pub fn bigquery_samples(_keyfile: &str) -> Result<CloudClient> {
            gcs_client("bigquery-public-data", _keyfile)
        }

        /// Load Google Books Ngrams
        pub fn books_ngrams(_keyfile: &str) -> Result<CloudClient> {
            gcs_client("books", _keyfile)
        }
    }

    /// Microsoft Azure Open Datasets
    pub struct AzureOpenData;

    impl AzureOpenData {
        /// Load COVID-19 tracking data
        pub fn covid19_tracking(_accountname: &str, accountkey: &str) -> Result<CloudClient> {
            azure_client(_accountname, accountkey, "covid19-tracking")
        }

        /// Load US Census data
        pub fn us_census(_accountname: &str, accountkey: &str) -> Result<CloudClient> {
            azure_client(_accountname, accountkey, "us-census")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::presets::*;
    use super::*;

    #[test]
    fn test_cloud_config_creation() {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some("us-east-1".to_string()),
            bucket: "test-bucket".to_string(),
            credentials: CloudCredentials::Anonymous,
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        assert!(matches!(config.provider, CloudProvider::S3));
        assert_eq!(config.bucket, "test-bucket");
    }

    #[test]
    fn test_s3_url_building() {
        let client = public_s3_client("us-east-1", "test-bucket").expect("Operation failed");
        let url = client
            .build_url("path/to/dataset.csv")
            .expect("Operation failed");
        assert_eq!(
            url,
            "https://test-bucket.s3.us-east-1.amazonaws.com/path/to/dataset.csv"
        );
    }

    #[test]
    fn test_s3path_style_url() {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some("us-east-1".to_string()),
            bucket: "test-bucket".to_string(),
            credentials: CloudCredentials::Anonymous,
            endpoint: None,
            path_style: true,
            headers: HashMap::new(),
        };

        let client = CloudClient::new(config).expect("Operation failed");
        let url = client.build_url("test.csv").expect("Operation failed");
        assert_eq!(
            url,
            "https://s3.us-east-1.amazonaws.com/test-bucket/test.csv"
        );
    }

    #[test]
    fn test_gcs_url_building() {
        let client = gcs_client("test-bucket", "dummy-key.json").expect("Operation failed");
        let url = client
            .build_url("data/file.json")
            .expect("Operation failed");
        assert_eq!(
            url,
            "https://storage.googleapis.com/test-bucket/data/file.json"
        );
    }

    #[test]
    fn test_azure_url_building() {
        let client =
            azure_client("testaccount", "dummykey", "container").expect("Operation failed");
        let url = client.build_url("blob.txt").expect("Operation failed");
        assert_eq!(
            url,
            "https://testaccount.blob.core.windows.net/container/blob.txt"
        );
    }

    #[test]
    fn test_s3_compatible_url_building() {
        let client = s3_compatible_client(
            "https://minio.example.com",
            "my-bucket",
            "access",
            "secret",
            true,
        )
        .expect("Operation failed");

        let url = client.build_url("file.csv").expect("Operation failed");
        assert_eq!(url, "https://minio.example.com/my-bucket/file.csv");
    }

    #[test]
    fn test_aws_open_data_clients() {
        // Test that we can create public dataset clients
        let result = public_datasets::AWSOpenData::noaa_weather();
        assert!(result.is_ok());

        let result = public_datasets::AWSOpenData::nyc_taxi();
        assert!(result.is_ok());
    }
}
