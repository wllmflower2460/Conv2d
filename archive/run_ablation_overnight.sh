#!/bin/bash
# Launch script for overnight ablation study
# Run with: nohup bash run_ablation_overnight.sh &

echo "=================================================="
echo "LAUNCHING OVERNIGHT ABLATION STUDY"
echo "Start time: $(date)"
echo "=================================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set CUDA settings for stability
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create results directory
RESULTS_DIR="ablation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Run ablation study with output logging
echo "Running ablation study..."
echo "Results will be saved to: $RESULTS_DIR"
echo "Log file: $RESULTS_DIR/ablation.log"

python -u experiments/ablation_overnight.py 2>&1 | tee $RESULTS_DIR/ablation.log

echo ""
echo "=================================================="
echo "ABLATION STUDY COMPLETE"
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "=================================================="

# Send notification (optional - uncomment if you have mail configured)
# echo "Ablation study complete. Results in $RESULTS_DIR" | mail -s "Ablation Complete" your-email@example.com