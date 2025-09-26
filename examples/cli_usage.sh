#!/bin/bash
# Conv2d CLI Usage Examples
# =========================

# Install the CLI
pip install -e .

# Show help
conv2d --help

# Show version
conv2d --version

# Enable verbose logging
conv2d --verbose preprocess data/raw data/processed

# 1. PREPROCESSING
# ----------------
# Basic preprocessing with default settings
conv2d preprocess data/raw data/processed

# Custom window and stride
conv2d preprocess data/raw data/processed --window-size 150 --stride 75

# With configuration file
conv2d preprocess data/raw data/processed --config conf/cli_example.yaml

# Disable quality checks (not recommended)
conv2d preprocess data/raw data/processed --no-check-quality

# 2. TRAINING
# -----------
# Train Conv2d-FSQ model
conv2d train data/processed models/conv2d_fsq --arch conv2d-fsq --epochs 100

# Train with custom hyperparameters
conv2d train data/processed models/conv2d_vq \
    --arch conv2d-vq \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.0005

# Train TCN-VAE baseline
conv2d train data/processed models/tcn_vae --arch tcn-vae

# 3. FSQ ENCODING
# ---------------
# Extract FSQ codes with default levels
conv2d fsq-encode models/conv2d_fsq/conv2d-fsq_best.pth data/processed codes/fsq_codes.pkl

# Custom quantization levels (8x8x8 = 512 codes)
conv2d fsq-encode models/conv2d_fsq/conv2d-fsq_best.pth data/processed codes/fsq_512.pkl --levels 8,8,8

# 4. CLUSTERING
# -------------
# K-means clustering
conv2d cluster codes/fsq_codes.pkl clusters/kmeans --n-clusters 12

# GMM clustering with custom support
conv2d cluster codes/fsq_codes.pkl clusters/gmm \
    --method gmm \
    --n-clusters 15 \
    --min-support 0.01

# Spectral clustering
conv2d cluster codes/fsq_codes.pkl clusters/spectral --method spectral

# 5. TEMPORAL SMOOTHING
# ---------------------
# Apply smoothing with defaults
conv2d smooth clusters/kmeans smoothed/kmeans

# Custom smoothing parameters
conv2d smooth clusters/kmeans smoothed/kmeans_custom \
    --window 9 \
    --min-duration 5

# 6. EVALUATION
# -------------
# Full evaluation with all metrics
conv2d eval models/conv2d_fsq data/test evaluation/full

# Basic metrics only
conv2d eval models/conv2d_fsq data/test evaluation/basic --metrics basic

# Extended behavioral analysis
conv2d eval models/conv2d_fsq data/test evaluation/extended --metrics extended

# 7. DEPLOYMENT PACKAGING
# -----------------------
# Package for ONNX deployment
conv2d pack models/conv2d_fsq deployment/conv2d_fsq.tar.gz

# Include evaluation results
conv2d pack models/conv2d_fsq deployment/conv2d_fsq_eval.tar.gz --eval evaluation/full

# Package for CoreML
conv2d pack models/conv2d_fsq deployment/conv2d_fsq_coreml.tar.gz --format coreml

# Package for Hailo without compression
conv2d pack models/conv2d_fsq deployment/conv2d_fsq_hailo.tar \
    --format hailo \
    --no-compress

# PIPELINE EXAMPLES
# =================

# Complete pipeline from raw data to deployment
echo "Running complete Conv2d pipeline..."

# Step 1: Preprocess
conv2d preprocess data/raw data/processed --config conf/cli_example.yaml

# Step 2: Train model
conv2d train data/processed models/conv2d_fsq --arch conv2d-fsq --epochs 100

# Step 3: Extract FSQ codes
conv2d fsq-encode models/conv2d_fsq/conv2d-fsq_best.pth data/processed codes/fsq.pkl

# Step 4: Cluster behaviors
conv2d cluster codes/fsq.pkl clusters/behaviors --n-clusters 12

# Step 5: Smooth sequences
conv2d smooth clusters/behaviors smoothed/behaviors

# Step 6: Evaluate performance
conv2d eval models/conv2d_fsq data/test evaluation/results

# Step 7: Package for deployment
conv2d pack models/conv2d_fsq deployment/package.tar.gz --eval evaluation/results

echo "Pipeline complete!"

# CHECK EXIT CODES
# =================
# The CLI uses specific exit codes for different failures:
# 0 - Success
# 1 - General error
# 2 - Data quality failure
# 3 - Model convergence failure
# 4 - Configuration error
# 5 - Deployment check failure

# Example: Check data quality and handle failure
conv2d preprocess data/raw data/processed
if [ $? -eq 2 ]; then
    echo "Data quality check failed! Please review the data."
    exit 1
fi

# Example: Ensure model converged
conv2d train data/processed models/test --epochs 50
if [ $? -eq 3 ]; then
    echo "Model failed to converge. Try adjusting hyperparameters."
    exit 1
fi