#!/bin/bash
# Test FSQ M1.3 Model Performance

MODEL_NAME="fsq_m13_behavioral_analysis"
HEF_FILE="${MODEL_NAME}.hef"

echo "=== FSQ M1.3 Performance Test ==="

if [ ! -f "$HEF_FILE" ]; then
    echo "Error: HEF file not found. Run ./compile_hailo8.sh first"
    exit 1
fi

# Latency test
echo "Testing inference latency..."
hailo run "$HEF_FILE" \
    --measure-latency \
    --measure-fps \
    --batch-size 1 \
    --num-iterations 1000

# Throughput test
echo ""
echo "Testing throughput..."
hailo run "$HEF_FILE" \
    --measure-fps \
    --batch-size 8 \
    --num-iterations 100

echo ""
echo "=== Performance Summary ==="
echo "Expected on Hailo-8:"
echo "- Latency: <15ms"
echo "- Throughput: >100 FPS"
echo "- Accuracy: 99.73%"
