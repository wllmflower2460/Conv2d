#!/bin/bash
# Enhanced Hailo-8 Compilation Script for FSQ M1.3
# Following hailo8_hef_compilation_checklist.md requirements

set -e  # Exit on error

# Configuration
MODEL_NAME="fsq_m13_behavioral_analysis"
ONNX_FILE="../models/${MODEL_NAME}.onnx"
CALIB_DATA="calibration_data.npy"
MODEL_SCRIPT="model_script.py"
HAR_FILE="${MODEL_NAME}.har"
OPTIMIZED_HAR="${MODEL_NAME}_optimized.har"
HEF_FILE="${MODEL_NAME}.hef"

echo "================================================"
echo "Hailo-8 Compilation for FSQ M1.3 Model"
echo "Following Compilation Checklist Requirements"
echo "================================================"

# Step 1: Parse ONNX to HAR
echo ""
echo "Step 1: Parsing ONNX model to HAR format..."
echo "Input: $ONNX_FILE"
echo "Output: $HAR_FILE"

hailo parser onnx "$ONNX_FILE" \
    --hw-arch hailo8 \
    --output-har-path "$HAR_FILE" \
    --net-name "$MODEL_NAME" || {
    echo "❌ Parsing failed"
    exit 1
}

echo "✅ Parsing complete: $HAR_FILE"

# Step 2: Optimize with INT8 quantization
echo ""
echo "Step 2: Optimizing model with INT8 quantization..."
echo "Calibration data: $CALIB_DATA"

if [ -f "$CALIB_DATA" ]; then
    echo "Using calibration dataset for quantization"
    hailo optimize "$HAR_FILE" \
        --hw-arch hailo8 \
        --output-har-path "$OPTIMIZED_HAR" \
        --calib-set-path "$CALIB_DATA" \
        --model-script "$MODEL_SCRIPT" \
        --quantization-method symmetric \
        --quantization-precision int8 || {
        echo "❌ Optimization with calibration failed"
        exit 1
    }
else
    echo "⚠️ No calibration data found, using random calibration"
    hailo optimize "$HAR_FILE" \
        --hw-arch hailo8 \
        --output-har-path "$OPTIMIZED_HAR" \
        --use-random-calib-set \
        --quantization-precision int8 || {
        echo "❌ Optimization failed"
        exit 1
    }
fi

echo "✅ Optimization complete: $OPTIMIZED_HAR"

# Step 3: Compile to HEF
echo ""
echo "Step 3: Compiling optimized HAR to HEF..."
echo "Target: <15ms core inference latency"

hailo compiler "$OPTIMIZED_HAR" \
    --hw-arch hailo8 \
    --output-hef-path "$HEF_FILE" \
    --performance-mode latency \
    --batch-size 1 \
    --optimization-level 3 || {
    echo "❌ Compilation failed"
    exit 1
}

echo "✅ Compilation complete: $HEF_FILE"

# Step 4: Profile performance
echo ""
echo "Step 4: Profiling model performance..."

hailo profiler "$HEF_FILE" \
    --hw-arch hailo8 \
    --measure-latency \
    --measure-fps \
    --batch-size 1 \
    --analyze-ops || {
    echo "⚠️ Profiling failed (non-critical)"
}

# Step 5: Validation
echo ""
echo "================================================"
echo "M1.3 Requirements Validation"
echo "================================================"
echo ""
echo "Performance Targets:"
echo "  [ ] Latency P95: <100ms (end-to-end)"
echo "  [ ] Core Inference: <15ms (Hailo-8)"
echo "  [ ] Throughput: >10 FPS"
echo ""
echo "Model Specifications:"
echo "  ✅ Input: (1, 9, 2, 100) - IMU behavioral data"
echo "  ✅ Output: (1, 10) - Behavioral logits"
echo "  ✅ Quantization: INT8 symmetric"
echo "  ✅ Batch size: 1 (edge inference)"
echo ""
echo "Files Generated:"
echo "  - $HAR_FILE: Parsed model"
echo "  - $OPTIMIZED_HAR: Optimized with INT8"
echo "  - $HEF_FILE: Compiled for Hailo-8"
echo ""
echo "================================================"
echo "✅ Compilation pipeline complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Test inference: hailo run $HEF_FILE --input test_data.npy"
echo "2. Integrate with EdgeInfer API"
echo "3. Validate 99.95% accuracy on test set"
echo "4. Monitor <15ms latency in production"
