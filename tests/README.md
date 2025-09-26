# Conv2d Test Suite

## Setup

The test suite requires the Conv2d package to be installed in development mode. This approach is the recommended way to handle imports in Python projects.

### Installation

From the project root directory:

```bash
# Install the package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run only integration tests
pytest -m integration

# Run with coverage report
pytest --cov=models --cov=preprocessing --cov-report=html

# Run specific test file
pytest tests/integration/test_fsq_pipeline.py

# Run with verbose output
pytest -v
```

## Test Structure

```
tests/
├── __init__.py
├── integration/
│   ├── __init__.py
│   └── test_fsq_pipeline.py    # Full pipeline integration tests
├── unit/
│   ├── __init__.py
│   ├── test_models.py           # Model unit tests
│   └── test_preprocessing.py    # Preprocessing unit tests
└── e2e/
    ├── __init__.py
    └── test_deployment.py        # End-to-end deployment tests
```

## Why No sys.path.append?

Using `sys.path.append` is considered an anti-pattern because:

1. **Package Structure**: With proper package structure and `setup.py`, imports work naturally
2. **Reproducibility**: `pip install -e .` ensures consistent environment across developers
3. **IDE Support**: IDEs can properly resolve imports and provide autocomplete
4. **CI/CD**: Same installation process works in CI/CD pipelines
5. **Best Practice**: Follows Python packaging standards (PEP 517/518)

## Test Markers

- `@pytest.mark.integration` - Integration tests (may be slower)
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Tests taking >1 second
- `@pytest.mark.gpu` - Tests requiring GPU

## Configuration

Test configuration is in `pytest.ini` at the project root. Key settings:

- `pythonpath = .` - Adds project root to Python path
- `testpaths = tests` - Default test directory
- Coverage targets: 70% minimum
- Automatic coverage reporting

## Troubleshooting

If imports fail:

1. Ensure you've installed the package: `pip install -e .`
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Verify package structure: `pip show conv2d-behavioral-sync`
4. Clear Python cache: `find . -type d -name __pycache__ -exec rm -r {} +`