#!/bin/bash
# Hailo-8 Compilation Script for FSQ M1.3 Model

set -e

MODEL_NAME="fsq_m13_behavioral_analysis"
ONNX_FILE="../models/${MODEL_NAME}.onnx"
HAR_FILE="${MODEL_NAME}.har"
OPTIMIZED_HAR="${MODEL_NAME}_optimized.har"
HEF_FILE="${MODEL_NAME}.hef"

echo "=== Hailo-8 Compilation for FSQ M1.3 ==="
echo "Model: $MODEL_NAME"
echo "ONNX: $ONNX_FILE"

# Check if ONNX file exists
if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: ONNX file not found: $ONNX_FILE"
    exit 1
fi

# Parse ONNX model
echo "Step 1: Parsing ONNX model..."
hailo parser onnx "$ONNX_FILE" \
    --hw-arch hailo8 \
    --output-har-path "$HAR_FILE" \
    --net-name "$MODEL_NAME"

echo "✅ Parsing complete: $HAR_FILE"

# Optimize model with quantization
echo "Step 2: Optimizing model..."
hailo optimize "$HAR_FILE" \
    --hw-arch hailo8 \
    --output-har-path "$OPTIMIZED_HAR" \
    --use-random-calib-set \
    --quantization-precision int8

echo "✅ Optimization complete: $OPTIMIZED_HAR"

# Compile to HEF
echo "Step 3: Compiling to HEF..."
hailo compiler "$OPTIMIZED_HAR" \
    --hw-arch hailo8 \
    --output-hef-path "$HEF_FILE"

echo "✅ Compilation complete: $HEF_FILE"

# Profile performance
echo "Step 4: Profiling performance..."
hailo profiler "$HEF_FILE" \
    --hw-arch hailo8 \
    --measure-latency \
    --measure-fps \
    --batch-size 1

echo ""
echo "=== Compilation Summary ==="
echo "Input:  $ONNX_FILE"
echo "Output: $HEF_FILE"
echo "Target: <15ms inference on Hailo-8"
echo "Ready for deployment!"
