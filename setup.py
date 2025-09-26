"""Setup script for Conv2d behavioral synchrony analysis package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="conv2d-behavioral-sync",
    version="1.0.0",
    author="Movement Team",
    author_email="movement@behavioral-sync.io",
    description="Conv2d-FSQ behavioral synchrony analysis with uncertainty quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/movement/conv2d-behavioral-sync",
    packages=find_packages(exclude=["tests", "tests.*", "archive", "archive.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "jidt": [
            "jpype1>=1.4.0",
        ],
        "hailo": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "conv2d-train=conv2d.scripts.train:main",
            "conv2d-evaluate=conv2d.scripts.evaluate:main",
            "conv2d-export=conv2d.scripts.export:main",
        ],
    },
    include_package_data=True,
    package_data={
        "conv2d": ["configs/*.yaml", "configs/*.yml"],
    },
)