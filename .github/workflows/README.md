# SciRS2 CI/CD Infrastructure

This directory contains comprehensive CI/CD workflows for the SciRS2 scientific computing workspace.

## 🚀 Workflow Overview

### Core Workflows

#### `workspace_ci.yml` - Main CI/CD Pipeline
- **Triggers**: Push to main branches, PRs, daily schedule
- **Features**:
  - Cross-platform build matrix (Linux, macOS, Windows)
  - Multiple Rust versions (stable, beta)
  - Feature matrix testing (default, full-features, minimal)
  - Cross-module integration testing
  - Code quality checks (clippy, rustfmt)
  - Documentation validation
  - Security auditing
  - Performance benchmarking
  - Release coordination

#### `nightly.yml` - Extended Testing
- **Triggers**: Daily at 2 AM UTC, manual dispatch
- **Features**:
  - Extended test suite with edge cases
  - Fuzz testing for critical modules
  - Cross-compilation matrix validation
  - Code coverage reporting
  - Memory leak detection
  - Documentation validation
  - External tool integration
  - Enhanced security auditing

#### `pypi-publish.yml` - PyPI Publishing
- **Triggers**: Manual workflow dispatch, release tags (`v*`)
- **Features**:
  - Cross-platform wheel building (Linux x86_64/aarch64, macOS x86_64/arm64, Windows x64)
  - Source distribution (sdist) generation
  - Automated publishing to TestPyPI or PyPI
  - Trusted publishing with OIDC authentication
  - Skip existing packages on republish
  - Maturin-based build system for PyO3 bindings

### Module-Specific Workflows

Individual modules have their own specialized workflows:

- **`scirs2-core`**: Cross-platform validation
- **`scirs2-optim`**: Release automation, security audit, memory leak detection
- **`scirs2-special`**: Performance regression monitoring
- **`scirs2-graph`**: Cross-platform testing

## 📋 Workflow Features

### Build Matrix Testing
- **Platforms**: Ubuntu, macOS, Windows
- **Rust Versions**: stable, beta, nightly (for extended tests)
- **Features**: default, full-features, minimal, module-specific combinations
- **Cross-compilation**: ARM, x86_64, WASM, musl targets

### Quality Assurance
- **Code Quality**: Clippy with zero warnings policy, rustfmt
- **Documentation**: Missing docs check, broken links detection
- **Security**: Cargo audit, dependency scanning, unsafe code detection
- **Performance**: Benchmarking, regression detection, memory profiling

### Integration Testing
- Cross-module compatibility validation
- End-to-end workflow testing
- External tool integration (Python, SciPy)
- API compatibility verification

### Release Management
- Version consistency validation
- Automated changelog checking
- Cross-platform artifact generation
- Security audit before release
- Performance validation

## 🔧 Setup Requirements

### GitHub Secrets
No secrets required for basic functionality. Optional secrets for enhanced features:

- `CODECOV_TOKEN`: For coverage reporting
- Custom tokens for notifications or external integrations

#### PyPI Publishing Setup
For PyPI publishing, configure **Trusted Publishing** (OIDC, no API tokens needed):

1. **PyPI Trusted Publishing Configuration**:
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher for `scirs2`
   - Repository: `cool-japan/scirs`
   - Workflow: `pypi-publish.yml`
   - Environment: `pypi`

2. **TestPyPI Trusted Publishing Configuration** (for testing):
   - Go to https://test.pypi.org/manage/account/publishing/
   - Add a new publisher for `scirs2`
   - Repository: `cool-japan/scirs`
   - Workflow: `pypi-publish.yml`
   - Environment: `testpypi`

3. **GitHub Environment Configuration**:
   - Go to repository Settings → Environments
   - Create environment: `pypi` (production)
   - Create environment: `testpypi` (testing)
   - Recommended: Add protection rules for `pypi` environment

### Repository Settings

#### Branch Protection
Recommended settings for `master` branch:
- Require status checks: `workspace-build`, `integration-tests`, `code-quality`
- Require up-to-date branches
- Include administrators

#### Actions Permissions
- Allow GitHub Actions to create and approve pull requests
- Allow GitHub Actions to write to repository

### Local Development Setup

#### Required Tools
```bash
# Core tools
cargo install cargo-nextest  # Faster test runner
cargo install cargo-audit    # Security auditing
cargo install cargo-deny     # License and dependency policy

# Documentation
cargo install mdbook         # Book generation
cargo install cargo-readme   # README generation

# Coverage (optional)
cargo install cargo-llvm-cov # Coverage reporting

# Development tools
cargo install cargo-watch    # File watching
cargo install cargo-expand   # Macro expansion
```

#### Pre-commit Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📊 Monitoring and Reporting

### Artifacts Generated
- **Build Artifacts**: Cross-platform binaries
- **Documentation**: API docs, coverage reports
- **Performance**: Benchmark results, regression analysis
- **Security**: Audit reports, vulnerability assessments
- **Quality**: Test reports, clippy analysis

### Dashboard Integration
Workflows generate structured data for:
- GitHub Checks API
- Status badges in README
- Performance tracking over time
- Security monitoring

## 🚨 Troubleshooting

### Common Issues

#### Build Failures
1. **Dependency Issues**: Check for version conflicts in `Cargo.lock`
2. **Platform Differences**: Review platform-specific features
3. **Memory Limits**: Increase timeout for large builds

#### Test Failures
1. **Flaky Tests**: Check for race conditions, use proper synchronization
2. **Platform Differences**: Use conditional compilation for platform-specific tests
3. **Resource Limits**: Ensure tests clean up properly

#### Performance Regressions
1. **Baseline Updates**: Update performance baselines after intentional changes
2. **Environmental Factors**: Account for CI runner variations
3. **Measurement Accuracy**: Use multiple runs for statistical significance

### Debugging Workflows

#### Local Reproduction
```bash
# Reproduce CI environment locally
cargo nextest run --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
```

#### CI Debugging
- Use `workflow_dispatch` for manual testing
- Add debug prints with `echo` in workflow steps
- Upload intermediate artifacts for inspection

## 🔄 Maintenance

### Regular Tasks
- Update tool versions in workflows (monthly)
- Review security audit results (weekly)
- Update performance baselines (after major changes)
- Clean up old artifacts (automated)

### Version Updates
When updating Rust or tool versions:
1. Test in feature branch first
2. Update all relevant workflows consistently
3. Update documentation and README
4. Monitor for regressions after merge

## 📚 Best Practices

### Workflow Design
- Keep workflows focused and modular
- Use matrix builds for comprehensive coverage
- Cache dependencies for faster builds
- Fail fast for quick feedback

### Security
- Minimize use of secrets
- Validate all external inputs
- Use official actions from trusted sources
- Regular security audits

### Performance
- Use `cargo-nextest` for faster test execution
- Implement incremental builds where possible
- Cache artifacts appropriately
- Monitor workflow execution times

## 🐍 PyPI Publishing Workflow

### Publishing to TestPyPI (Testing)
```bash
# Manual trigger via GitHub Actions UI:
# 1. Go to Actions → PyPI Publish → Run workflow
# 2. Select publish_target: testpypi
# 3. Click "Run workflow"

# Or trigger via GitHub CLI:
gh workflow run pypi-publish.yml -f publish_target=testpypi
```

### Publishing to PyPI (Production)
```bash
# Option 1: Tag-based automatic publishing
git tag v0.1.5
git push origin v0.1.5

# Option 2: Manual workflow dispatch
gh workflow run pypi-publish.yml -f publish_target=pypi -f version_tag=v0.1.5
```

### Build-Only (No Publishing)
```bash
# Test wheel building without publishing
gh workflow run pypi-publish.yml -f publish_target=none
```

### Installation Testing
```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ scirs2

# Install from PyPI
pip install scirs2
```

### Pre-Release Checklist
Before publishing to PyPI:
- [ ] Update version in `scirs2-python/Cargo.toml`
- [ ] Update version in `scirs2-python/pyproject.toml`
- [ ] Update CHANGELOG.md with release notes
- [ ] Run local tests: `cd scirs2-python && cargo test`
- [ ] Test Python bindings: `cd scirs2-python && maturin develop && pytest`
- [ ] Build locally: `cd scirs2-python && maturin build --release`
- [ ] Test on TestPyPI first
- [ ] Create and push version tag
- [ ] Verify PyPI deployment
- [ ] Update documentation with new version

## 🎯 Future Enhancements

### Planned Features
- [ ] Automated dependency updates (Dependabot)
- [ ] Integration with external benchmarking services
- [ ] Enhanced notification system
- [ ] Performance regression alerts
- [ ] Automated security scanning
- [x] Integration with package registries (PyPI via Trusted Publishing)

### Contributing
See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Adding new workflows
- Modifying existing CI/CD
- Testing workflow changes
- Best practices for CI maintenance