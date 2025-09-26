# Conv2d-FSQ-HSMM Package Structure

## 🎯 Modern Python Package Setup

This project now follows modern Python packaging best practices with full type safety, pre-commit hooks, and clean separation between library and CLI code.

## 📦 Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/conv2d-fsq-hsmm.git
cd conv2d-fsq-hsmm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Production Installation

```bash
# Install from PyPI (when published)
pip install conv2d-fsq-hsmm

# Or install from GitHub
pip install git+https://github.com/yourusername/conv2d-fsq-hsmm.git
```

## 🏗️ Package Structure

```
Conv2d/
├── src/                        # Source code (clean imports)
│   └── conv2d/                 # Main package
│       ├── __init__.py         # Public API exports
│       ├── py.typed            # PEP 561 type marker
│       ├── cli/                # CLI commands (isolated)
│       │   ├── main.py         # Main entry point
│       │   ├── train.py        # Training command
│       │   ├── evaluate.py     # Evaluation command
│       │   └── export.py       # Export command
│       ├── models/             # Model implementations
│       │   ├── fsq_layer.py    # FSQ quantization
│       │   ├── hsmm.py         # HSMM components
│       │   └── integrated.py   # Complete model
│       ├── preprocessing/      # Data preprocessing
│       │   ├── data_quality.py # NaN handling
│       │   ├── features.py     # Feature extraction
│       │   └── vectorized_ops.py # Optimized ops
│       ├── training/           # Training utilities
│       │   ├── config.py       # Configuration
│       │   └── trainer.py      # Training loop
│       └── utils/              # Utilities
│           ├── logging.py      # Logging setup
│           └── seed.py         # Reproducibility
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml             # Modern Python packaging
├── .pre-commit-config.yaml    # Code quality hooks
└── README.md                   # Project documentation
```

## 🛠️ Development Tools

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

**Included hooks:**
- **Ruff**: Fast Python linter (replaces flake8, pylint, isort)
- **Black**: Code formatter
- **isort**: Import sorting
- **mypy**: Static type checking (strict mode)
- **docformatter**: Docstring formatting
- **bandit**: Security checks
- **prettier**: YAML/JSON/Markdown formatting
- **detect-secrets**: Prevent secrets in code

### Type Checking

All code includes type hints with `from __future__ import annotations`:

```python
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from torch import Tensor

def process_data(
    data: NDArray[np.float32],
    threshold: Optional[float] = None,
) -> Tuple[Tensor, dict[str, float]]:
    """Process data with full type safety."""
    ...
```

Run type checking:
```bash
mypy src/conv2d --strict
```

### Testing

```bash
# Run tests with coverage
pytest --cov=conv2d --cov-report=html

# Run specific test markers
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests

# Parallel testing
pytest -n auto  # Use all CPU cores
```

## 💻 CLI Usage

The package provides a unified CLI interface:

```bash
# Training
conv2d train --config configs/training.yaml

# Evaluation
conv2d evaluate --checkpoint models/best.pth --batch-size 64

# Export for deployment
conv2d export --checkpoint models/best.pth --format onnx --optimize

# Get help
conv2d --help
conv2d train --help
```

## 🐍 Python API Usage

```python
from conv2d import (
    Conv2dFSQHSMM,
    DataQualityHandler,
    KinematicFeatureExtractor,
    Trainer,
    TrainingConfig,
    set_seed,
)

# Set seed for reproducibility
set_seed(42)

# Initialize model
model = Conv2dFSQHSMM(
    input_channels=9,
    fsq_levels=[8, 6, 5],
    num_states=32,
)

# Set up training
config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=100,
)

trainer = Trainer(model, config)

# Train model
trainer.train(train_loader, val_loader)
```

## 📋 Configuration Files

### pyproject.toml Features

- **Build system**: Modern setuptools with PEP 517/518
- **Dependencies**: Explicit version constraints
- **Optional dependencies**: dev, hailo, coreml, docs groups
- **Tool configs**: Ruff, Black, isort, mypy, pytest, coverage
- **Entry points**: CLI commands automatically installed

### Pre-commit Configuration

- **Automatic fixes**: Trailing whitespace, EOF, import sorting
- **Code quality**: Ruff linting, Black formatting
- **Type safety**: mypy with strict mode
- **Security**: Bandit scanning, secret detection
- **CI integration**: Pre-commit.ci support

## 🚀 Building and Publishing

### Build Package

```bash
# Install build tools
pip install build twine

# Build distributions
python -m build

# Check distributions
twine check dist/*
```

### Publish to PyPI

```bash
# Test PyPI (optional)
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

## 🔧 Best Practices Implemented

1. **Package Structure**
   - `src/` layout prevents import conflicts
   - Separate CLI from library code
   - Clean public API in `__init__.py`

2. **Type Safety**
   - Full type hints with `from __future__ import annotations`
   - `py.typed` marker for PEP 561
   - Strict mypy configuration
   - numpy.typing and torch type stubs

3. **Code Quality**
   - Pre-commit hooks run automatically
   - Ruff replaces multiple linters
   - Black ensures consistent formatting
   - isort manages imports

4. **Testing**
   - pytest with plugins (cov, xdist, timeout)
   - Test markers for organization
   - Parallel test execution
   - Coverage reporting

5. **Documentation**
   - Type hints serve as inline docs
   - Docstrings follow Google style
   - README with examples
   - Sphinx-ready structure

## 📝 Migration Notes

To migrate existing code to the new structure:

1. Move files to `src/conv2d/` structure
2. Update imports to use absolute paths: `from conv2d.models import FSQLayer`
3. Add `from __future__ import annotations` to all files
4. Add type hints to function signatures
5. Run `pre-commit run --all-files` to fix formatting
6. Run `mypy src/conv2d` to check types

## 🔗 Resources

- [Python Packaging Guide](https://packaging.python.org)
- [PEP 517/518](https://www.python.org/dev/peps/pep-0517/) - Build system
- [PEP 561](https://www.python.org/dev/peps/pep-0561/) - Type hints
- [Pre-commit](https://pre-commit.com) - Git hooks
- [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter