# Contributing to medrs

Thank you for your interest in contributing to medrs! This guide will help you get started.

## Quick Start

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/medrs.git
   cd medrs
   ```

2. **Set up the development environment**
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Install Python dependencies
   pip install -e ".[dev]"

   # Build the Python bindings
   maturin develop --features python
   ```

3. **Run the test suite**
   ```bash
   # Rust tests (73 tests)
   cargo test

   # Python tests
   pytest tests/
   ```

## Development Workflow

### 1. Make Changes

- **Rust code**: Make changes in `src/`
- **Python code**: Make changes in `src/python/`
- **Examples**: Add examples in `examples/`
- **Tests**: Add tests in `tests/`
- **Documentation**: Update docs in `docs/`

### 2. Test Your Changes

```bash
# Run Rust tests
cargo test

# Run Rust tests with output
cargo test -- --nocapture

# Run specific Rust test
cargo test test_name

# Run Python tests
pytest tests/

# Run with coverage
pytest tests/ --cov=medrs

# Run benchmarks
cargo bench
python examples/basic/benchmark.py

# Test specific integrations
pytest tests/test_monai_integration.py
pytest tests/test_python_integration.py
```

### 3. Submit Pull Request

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

3. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
medrs/
+-- src/                     # Core Rust library
|   +-- nifti/              # NIfTI file format support
|   +-- transforms/         # High-performance transforms
|   +-- pipeline/           # Training pipeline utilities
|   +-- python.rs           # Python bindings (PyO3)
+-- src/python/             # Python API and utilities
|   +-- __init__.py         # Main Python module
|   +-- exceptions.py       # Error handling
|   +-- performance_profiler.py  # Performance monitoring
+-- examples/               # Usage examples
|   +-- basic/             # Getting started examples
|   +-- integrations/      # Framework examples
|   +-- advanced/          # Production patterns
+-- tests/                 # Comprehensive test suite
|   +-- integration/       # Integration tests
|   +-- conftest.py        # Test configuration
+-- benchmarks/            # Performance benchmarks
+-- docs/                  # Documentation
+-- pyproject.toml         # Python project configuration
```

## Testing Guidelines

### Writing Tests

1. **Unit Tests**: Test individual functions and modules
2. **Integration Tests**: Test cross-language functionality
3. **Performance Tests**: Ensure no performance regressions
4. **Property-Based Tests**: Use Hypothesis for edge cases

### Test Categories

```python
import pytest

@pytest.mark.unit
def test_specific_function():
    """Test a specific function."""
    pass

@pytest.mark.integration
def test_rust_python_integration():
    """Test Rust-Python integration."""
    pass

@pytest.mark.benchmark
def test_performance_benchmark():
    """Test performance critical path."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Test slow operation (marked as slow)."""
    pass
```

### Adding New Features

When adding new features:

1. **Add tests** for the new functionality
2. **Add examples** demonstrating usage
3. **Update documentation** with API changes
4. **Run benchmarks** to ensure performance
5. **Test integrations** with PyTorch/JAX/MONAI

## Code Style

### Rust Code

- Follow `rustfmt` formatting
- Use `clippy` for linting
- Write comprehensive doc comments
- Use type hints and proper error handling

```bash
# Format Rust code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

### Python Code

- Follow PEP 8 (use `black` formatter)
- Use type hints throughout
- Write comprehensive docstrings
- Use proper exception handling

```bash
# Format Python code
black src/python/ tests/ examples/

# Type checking
mypy src/python/
```

## Performance Guidelines

### Critical Paths

- **I/O Operations**: Must be as fast as possible
- **Memory Usage**: Minimize allocations
- **Transform Operations**: Use SIMD where possible

### Benchmarking

```bash
# Rust benchmarks
cargo bench

# Python benchmarks
pytest tests/ --benchmark-only

# Compare with alternatives
python examples/integrations/performance_comparison.py
```

### Performance Checklist

- [ ] Run benchmarks before and after changes
- [ ] No regressions in critical paths
- [ ] Memory usage remains optimal
- [ ] SIMD optimizations maintained

## Documentation

### API Documentation

- Document all public functions and classes
- Include parameter descriptions and examples
- Mention performance characteristics

```rust
/// Load a NIfTI file with crop-first optimization.
///
/// # Arguments
///
/// * `path` - Path to the NIfTI file
/// * `offset` - Starting voxel offset [x, y, z]
/// * `shape` - Desired output shape [x, y, z]
///
/// # Returns
///
/// `MedicalImage<T>` with loaded data
///
/// # Performance
///
/// This function reads only the required bytes, achieving 40x
/// memory reduction compared to loading the entire volume.
///
/// # Examples
///
/// ```rust
/// let img = medrs::nifti::load_cropped(
///     "volume.nii.gz",
///     [32, 32, 16],
///     [64, 64, 64]
/// )?;
/// ```
pub fn load_cropped<T: DataType>(path: &str, offset: [usize; 3], shape: [usize; 3]) -> Result<MedicalImage<T>> {
    // Implementation
}
```

### Examples

- Add examples for new features
- Include performance benchmarks
- Show integration with frameworks
- Provide real-world use cases

## Bug Reports

When reporting bugs:

1. **Use the bug report template**
2. **Include minimal reproduction example**
3. **Provide system information** (OS, Python version, Rust version)
4. **Include performance benchmarks** if relevant
5. **Attach test files** if possible

## Feature Requests

When requesting features:

1. **Describe the use case** in detail
2. **Explain why existing solutions don't work**
3. **Propose API design** if you have ideas
4. **Consider performance implications**

## Review Process

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] Performance is maintained
- [ ] No breaking changes without version bump
- [ ] Integration tests pass

### Getting Reviews

1. **Request review** from maintainers
2. **Address feedback** promptly
3. **Update tests** based on review
4. **Re-request review** after changes

## Version Management

### Semantic Versioning

- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes and improvements

### Release Process

1. **Update version numbers** in `Cargo.toml` and `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Tag the release**
4. **Build and publish to PyPI**

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and contribute

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Documentation**: For usage guidance

## Recognition

Contributors are recognized in:

- **README.md**: Major contributors section
- **CHANGELOG.md**: Feature attributions
- **Release notes**: Specific contributions
- **Documentation**: Author credits

## Release Checklist

Before releasing:

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Performance benchmarks run
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Examples tested
- [ ] Integration tests pass
- [ ] Security review (if applicable)

## Development Tools

### Useful Commands

```bash
# Development build
maturin develop --features python

# Release build
maturin build --release

# Run specific test
pytest tests/test_specific.py::test_function

# Run with specific marker
pytest tests/ -m "not slow"

# Coverage report
pytest tests/ --cov=medrs --cov-report=html

# Documentation build
cd docs && make html

# Performance profiling
cargo run --example profile_workload --release
```

### IDE Setup

#### VSCode

Recommended extensions:
- **Rust Analyzer**: Rust language support
- **Python**: Python language support
- **CodeLLDB**: Debugging support

#### PyCharm

- Install Rust plugin
- Configure Python interpreter
- Set up pytest configuration

---

Thank you for contributing to medrs! Your contributions help make medical imaging faster and more accessible for everyone. 