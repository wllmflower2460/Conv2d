#!/usr/bin/env python3
# Hailo Model Script for Conv2d-FSQ
# Auto-generated for M1.3 deployment

import numpy as np
from hailo_sdk_client import ClientRunner

def preprocess(images):
    '''Preprocess input for the model'''
    # Input shape: (batch, 9, 2, 100)
    # Already normalized in calibration dataset
    return images

def postprocess(outputs):
    '''Postprocess model outputs'''
    # Output: behavioral logits
    # Apply softmax for probabilities
    probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
    return probs

# Model configuration
model_config = {
    'input_shape': (1, 9, 2, 100),
    'output_shape': (1, 10),
    'quantization': 'int8',
    'optimization_level': 3,
    'batch_size': 1
}
