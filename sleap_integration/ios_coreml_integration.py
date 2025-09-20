#!/usr/bin/env python3
"""
iOS Core ML Integration for SLEAP Models
Converts SLEAP models to Core ML and generates Swift integration code

Usage:
  python ios_coreml_integration.py --model model.h5 --output-dir ios_models/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import sleap
    import tensorflow as tf
    import numpy as np
    import coremltools as ct
    from coremltools.models.utils import rename_feature
    from coremltools.models.datatypes import Array, Image
except ImportError as e:
    print(f"❌ Required packages not installed: {e}")
    print("Install with: pip install sleap coremltools tensorflow")
    sys.exit(1)

class iOSCoreMLIntegration:
    """Convert SLEAP models to Core ML and generate Swift integration code"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "neck", "left_front_shoulder", "left_front_elbow", "left_front_wrist", "left_front_paw",
            "right_front_shoulder", "right_front_elbow", "right_front_wrist", "right_front_paw",
            "spine_mid", "left_back_shoulder", "left_back_elbow", "left_back_wrist", "left_back_paw",
            "right_back_shoulder", "right_back_elbow", "right_back_wrist", "right_back_paw",
            "tail_base", "tail_mid", "tail_tip"
        ]
        
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 14),  # Head
            (5, 6), (6, 7), (7, 8), (8, 9),                    # Left front leg
            (5, 10), (10, 11), (11, 12), (12, 13),             # Right front leg
            (14, 15), (15, 16), (16, 17), (17, 18),            # Left back leg
            (14, 19), (19, 20), (20, 21), (21, 22),            # Right back leg
            (14, 23), (23, 24), (24, 25)                       # Tail
        ]
    
    def convert_sleap_to_coreml(self, model_path: str, model_name: str) -> Optional[str]:
        """Convert SLEAP model to Core ML format"""
        print(f"🔄 Converting SLEAP model to Core ML: {model_name}")
        
        try:
            # Load SLEAP model
            tf_model = tf.keras.models.load_model(model_path, compile=False)
            
            print(f"📋 Model info:")
            print(f"   • Input shape: {tf_model.input_shape}")
            print(f"   • Output shape: {tf_model.output_shape}")
            
            # Determine input size from model
            input_shape = tf_model.input_shape
            if len(input_shape) == 4:  # (batch, height, width, channels)
                height, width, channels = input_shape[1], input_shape[2], input_shape[3]
            else:
                height, width, channels = 384, 384, 3  # Default
            
            print(f"   • Processing size: {height}x{width}x{channels}")
            
            # Convert to Core ML
            coreml_model = ct.convert(
                tf_model,
                inputs=[ct.ImageType(
                    name="input_image",
                    shape=(1, height, width, channels),
                    bias=[-1, -1, -1],  # Normalize from [0,255] to [-1,1]
                    scale=2.0/255.0
                )],
                outputs=[ct.TensorType(name="heatmaps")],
                minimum_deployment_target=ct.target.iOS15,
                compute_precision=ct.precision.FLOAT16
            )
            
            # Add model metadata
            coreml_model.short_description = f"DataDogs Animal Pose Estimation - {model_name}"
            coreml_model.author = "DataDogs Platform"
            coreml_model.license = "Proprietary"
            coreml_model.version = "1.0.0"
            
            # Add input description
            coreml_model.input_description["input_image"] = "RGB image of animal (dog/cat) for pose estimation"
            coreml_model.output_description["heatmaps"] = f"Confidence heatmaps for {len(self.keypoint_names)} keypoints"
            
            # Save Core ML model
            coreml_path = self.output_dir / f"{model_name}.mlmodel"
            coreml_model.save(str(coreml_path))
            
            # Test the model
            self.test_coreml_model(str(coreml_path), height, width, channels)
            
            print(f"✅ Core ML model saved: {coreml_path}")
            return str(coreml_path)
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_coreml_model(self, model_path: str, height: int, width: int, channels: int):
        """Test Core ML model with dummy input"""
        try:
            import coremltools as ct
            
            # Load model
            model = ct.models.MLModel(model_path)
            
            # Create dummy input
            dummy_input = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
            
            # Make prediction
            prediction = model.predict({"input_image": dummy_input})
            
            output_shape = list(prediction.values())[0].shape
            print(f"✅ Model test successful - Output shape: {output_shape}")
            
        except Exception as e:
            print(f"⚠️  Model test failed: {e}")
    
    def generate_swift_integration_code(self, model_name: str, coreml_path: str) -> str:
        """Generate Swift code for iOS integration"""
        print(f"📝 Generating Swift integration code...")
        
        swift_code = f'''//
//  {model_name}PoseEstimator.swift
//  DataDogs
//
//  Custom SLEAP model integration for animal pose estimation
//  Generated automatically - do not modify
//

import Foundation
import CoreML
import Vision
import UIKit
import simd

// MARK: - Custom SLEAP Pose Estimator

@available(iOS 15.0, *)
class {model_name}PoseEstimator: ObservableObject {{
    
    // MARK: - Properties
    
    private var model: MLModel?
    private let modelName = "{model_name}"
    
    @Published var isModelLoaded = false
    @Published var lastError: Error?
    
    // MARK: - Keypoint Configuration
    
    enum KeypointType: String, CaseIterable {{
        {self._generate_keypoint_enum()}
        
        var index: Int {{
            return Self.allCases.firstIndex(of: self) ?? 0
        }}
        
        var displayName: String {{
            return self.rawValue.replacingOccurrences(of: "_", with: " ").capitalized
        }}
    }}
    
    struct DetectedKeypoint {{
        let type: KeypointType
        let location: CGPoint
        let confidence: Float
        let isVisible: Bool
        
        var normalizedLocation: CGPoint {{
            return location
        }}
    }}
    
    struct PoseResult {{
        let keypoints: [DetectedKeypoint]
        let boundingBox: CGRect
        let overallConfidence: Float
        let processingTime: TimeInterval
        let modelName: String
        
        var visibleKeypoints: [DetectedKeypoint] {{
            return keypoints.filter {{ $0.isVisible }}
        }}
        
        var confidenceScore: Float {{
            let visiblePoints = visibleKeypoints
            guard !visiblePoints.isEmpty else {{ return 0.0 }}
            return visiblePoints.map {{ $0.confidence }}.reduce(0, +) / Float(visiblePoints.count)
        }}
    }}
    
    // MARK: - Skeleton Connections
    
    static let skeletonConnections: [(Int, Int)] = [
        {self._generate_skeleton_connections()}
    ]
    
    // MARK: - Initialization
    
    init() {{
        loadModel()
    }}
    
    private func loadModel() {{
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") else {{
            print("❌ Model file not found: {{modelName}}.mlmodel")
            DispatchQueue.main.async {{
                self.lastError = NSError(domain: "ModelError", code: 1, 
                    userInfo: [NSLocalizedDescriptionKey: "Model file not found"])
            }}
            return
        }}
        
        do {{
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine when available
            
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            
            DispatchQueue.main.async {{
                self.isModelLoaded = true
                print("✅ {{self.modelName}} model loaded successfully")
            }}
        }} catch {{
            print("❌ Failed to load model: {{error}}")
            DispatchQueue.main.async {{
                self.lastError = error
            }}
        }}
    }}
    
    // MARK: - Pose Detection
    
    func detectPose(in pixelBuffer: CVPixelBuffer, completion: @escaping (PoseResult?) -> Void) {{
        guard let model = model else {{
            completion(nil)
            return
        }}
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Perform inference on background queue
        DispatchQueue.global(qos: .userInteractive).async {{
            do {{
                // Prepare input
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "input_image": MLFeatureValue(pixelBuffer: pixelBuffer)
                ])
                
                // Run prediction
                let prediction = try model.prediction(from: input)
                
                // Extract heatmaps
                guard let heatmaps = prediction.featureValue(for: "heatmaps")?.multiArrayValue else {{
                    print("❌ Failed to extract heatmaps")
                    DispatchQueue.main.async {{ completion(nil) }}
                    return
                }}
                
                // Process heatmaps to keypoints
                let keypoints = self.processHeatmaps(heatmaps, imageSize: CGSize(
                    width: CVPixelBufferGetWidth(pixelBuffer),
                    height: CVPixelBufferGetHeight(pixelBuffer)
                ))
                
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime
                
                // Create result
                let result = PoseResult(
                    keypoints: keypoints,
                    boundingBox: self.calculateBoundingBox(from: keypoints),
                    overallConfidence: self.calculateOverallConfidence(from: keypoints),
                    processingTime: processingTime,
                    modelName: self.modelName
                )
                
                DispatchQueue.main.async {{
                    completion(result)
                }}
                
            }} catch {{
                print("❌ Prediction failed: {{error}}")
                DispatchQueue.main.async {{
                    self.lastError = error
                    completion(nil)
                }}
            }}
        }}
    }}
    
    // MARK: - Heatmap Processing
    
    private func processHeatmaps(_ heatmaps: MLMultiArray, imageSize: CGSize) -> [DetectedKeypoint] {{
        var keypoints: [DetectedKeypoint] = []
        
        let heatmapShape = heatmaps.shape
        guard heatmapShape.count >= 3 else {{
            print("❌ Invalid heatmap shape: {{heatmapShape}}")
            return keypoints
        }}
        
        let numKeypoints = heatmapShape[0].intValue
        let heatmapHeight = heatmapShape[1].intValue
        let heatmapWidth = heatmapShape[2].intValue
        
        // Process each keypoint
        for keypointIndex in 0..<min(numKeypoints, KeypointType.allCases.count) {{
            let keypointType = KeypointType.allCases[keypointIndex]
            
            // Find peak in heatmap
            var maxConfidence: Float = 0.0
            var maxRow = 0
            var maxCol = 0
            
            for row in 0..<heatmapHeight {{
                for col in 0..<heatmapWidth {{
                    let index = keypointIndex * heatmapHeight * heatmapWidth + row * heatmapWidth + col
                    let confidence = heatmaps[index].floatValue
                    
                    if confidence > maxConfidence {{
                        maxConfidence = confidence
                        maxRow = row
                        maxCol = col
                    }}
                }}
            }}
            
            // Convert heatmap coordinates to image coordinates
            let normalizedX = Float(maxCol) / Float(heatmapWidth - 1)
            let normalizedY = Float(maxRow) / Float(heatmapHeight - 1)
            
            let imageX = CGFloat(normalizedX) * imageSize.width
            let imageY = CGFloat(normalizedY) * imageSize.height
            
            // Apply confidence threshold
            let confidenceThreshold: Float = 0.3
            let isVisible = maxConfidence > confidenceThreshold
            
            let keypoint = DetectedKeypoint(
                type: keypointType,
                location: CGPoint(x: imageX, y: imageY),
                confidence: maxConfidence,
                isVisible: isVisible
            )
            
            keypoints.append(keypoint)
        }}
        
        return keypoints
    }}
    
    private func calculateBoundingBox(from keypoints: [DetectedKeypoint]) -> CGRect {{
        let visibleKeypoints = keypoints.filter {{ $0.isVisible }}
        guard !visibleKeypoints.isEmpty else {{ return .zero }}
        
        let xCoordinates = visibleKeypoints.map {{ $0.location.x }}
        let yCoordinates = visibleKeypoints.map {{ $0.location.y }}
        
        let minX = xCoordinates.min() ?? 0
        let maxX = xCoordinates.max() ?? 0
        let minY = yCoordinates.min() ?? 0
        let maxY = yCoordinates.max() ?? 0
        
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }}
    
    private func calculateOverallConfidence(from keypoints: [DetectedKeypoint]) -> Float {{
        let visibleKeypoints = keypoints.filter {{ $0.isVisible }}
        guard !visibleKeypoints.isEmpty else {{ return 0.0 }}
        
        let totalConfidence = visibleKeypoints.reduce(0) {{ $0 + $1.confidence }}
        return totalConfidence / Float(visibleKeypoints.count)
    }}
    
    // MARK: - Integration with DataDogs Pipeline
    
    func shouldUploadToFirebase(_ result: PoseResult, threshold: Float = 0.6) -> Bool {{
        // Upload uncertain poses for distributed learning
        return result.overallConfidence < threshold
    }}
    
    func createFirebaseUploadData(_ result: PoseResult, imageData: Data?) -> [String: Any] {{
        var data: [String: Any] = [
            "model_name": result.modelName,
            "confidence": result.overallConfidence,
            "processing_time": result.processingTime,
            "timestamp": Date().timeIntervalSince1970,
            "keypoints": result.keypoints.map {{ keypoint in
                return [
                    "type": keypoint.type.rawValue,
                    "x": keypoint.location.x,
                    "y": keypoint.location.y,
                    "confidence": keypoint.confidence,
                    "visible": keypoint.isVisible
                ]
            }}
        ]
        
        if let imageData = imageData {{
            data["image_data"] = imageData.base64EncodedString()
        }}
        
        return data
    }}
}}

// MARK: - SwiftUI Integration

@available(iOS 15.0, *)
extension {model_name}PoseEstimator {{
    
    func detectPoseFromImage(_ image: UIImage, completion: @escaping (PoseResult?) -> Void) {{
        guard let pixelBuffer = image.pixelBuffer() else {{
            completion(nil)
            return
        }}
        
        detectPose(in: pixelBuffer, completion: completion)
    }}
}}

// MARK: - UIImage Extension

extension UIImage {{
    func pixelBuffer() -> CVPixelBuffer? {{
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {{
            return nil
        }}
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {{
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }}
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        context.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }}
}}
'''
        
        # Save Swift code
        swift_path = self.output_dir / f"{model_name}PoseEstimator.swift"
        with open(swift_path, 'w') as f:
            f.write(swift_code)
        
        print(f"✅ Swift integration code saved: {swift_path}")
        return str(swift_path)
    
    def _generate_keypoint_enum(self) -> str:
        """Generate Swift enum cases for keypoints"""
        cases = []
        for name in self.keypoint_names:
            case_name = name.replace(" ", "_").lower()
            cases.append(f'        case {case_name} = "{name}"')
        return "\\n".join(cases)
    
    def _generate_skeleton_connections(self) -> str:
        """Generate Swift array of skeleton connections"""
        connections = []
        for src, dst in self.skeleton_connections:
            connections.append(f"        ({src}, {dst})")
        return ",\\n".join(connections)
    
    def create_xcode_integration_guide(self, model_name: str, coreml_path: str, swift_path: str) -> str:
        """Create integration guide for Xcode"""
        guide_content = f"""# {model_name} Xcode Integration Guide

## Files Required

1. **Core ML Model**: `{Path(coreml_path).name}`
2. **Swift Integration**: `{Path(swift_path).name}`

## Integration Steps

### 1. Add Files to Xcode Project

1. Drag `{Path(coreml_path).name}` into your Xcode project
2. Ensure "Add to target" is checked for your main app target
3. Add `{Path(swift_path).name}` to your project

### 2. Update PoseEstimationView

Replace the existing Apple Vision framework code with custom model:

```swift
// In PoseEstimationViewModel.swift
@available(iOS 15.0, *)
class PoseEstimationViewModel: ObservableObject {{
    
    // Add custom model estimator
    private let customEstimator = {model_name}PoseEstimator()
    
    // Use both models for comparison
    @Published var useCustomModel = false
    
    func toggleModel() {{
        useCustomModel.toggle()
    }}
    
    // Modify existing detection method
    private func detectPoseInFrame(imageBuffer: CVImageBuffer) {{
        if useCustomModel {{
            customEstimator.detectPose(in: imageBuffer) {{ [weak self] result in
                guard let result = result else {{ return }}
                
                // Convert to existing AnimalPose format
                let animalPose = self?.convertCustomResultToAnimalPose(result)
                // ... existing processing logic
            }}
        }} else {{
            // Existing Apple Vision framework code
            performVisionRequest(on: imageBuffer)
        }}
    }}
}}
```

### 3. Enable A/B Testing

```swift
// In FeatureFlags.swift
struct FeatureFlags {{
    /// Use custom SLEAP model vs Apple Vision
    static let useCustomPoseModel = false
    
    /// A/B test percentage for custom model
    static let customModelRolloutPercentage = 25.0
}}

// In Firebase RemoteConfig
"use_custom_pose_model": false,
"custom_model_rollout_percentage": 25.0
```

### 4. Firebase Integration

The custom estimator includes built-in Firebase integration:

```swift
// Automatic uncertain pose upload
if customEstimator.shouldUploadToFirebase(result) {{
    let uploadData = customEstimator.createFirebaseUploadData(result, imageData: frameData)
    FirebaseTrainingManager.shared.uploadUncertainPose(data: uploadData)
}}
```

## Performance Comparison

### Apple Vision Framework
- ✅ Real-time performance (60 FPS)
- ✅ Optimized for iOS devices
- ✅ No additional model size
- ❌ Generic animal detection
- ❌ Limited customization

### Custom SLEAP Model
- ✅ Trained on your specific data
- ✅ 25 custom keypoints
- ✅ Improved accuracy on DataDogs dataset
- ❌ Larger app size (~50MB model)
- ❌ Potentially slower inference

## Testing Strategy

1. **Development**: Test both models side-by-side
2. **Staging**: A/B test with 25% custom model users
3. **Production**: Gradual rollout based on performance metrics

## Monitoring

Track these metrics via Firebase Analytics:
- Model inference time
- Detection confidence scores
- User satisfaction ratings
- Firebase upload frequency (uncertain poses)

## Troubleshooting

### Model Not Loading
- Verify `{Path(coreml_path).name}` is in app bundle
- Check iOS deployment target is 15.0+
- Ensure model file isn't corrupted

### Poor Performance
- Monitor memory usage during inference
- Consider reducing model input size
- Use background queues for processing

### Integration Issues
- Review Swift compilation errors
- Check iOS version compatibility
- Verify Core ML framework is linked

## Next Steps

1. **Immediate**: Basic integration and testing
2. **Short-term**: A/B test setup and monitoring
3. **Long-term**: Model updates based on Firebase training data
"""
        
        guide_path = self.output_dir / f"{model_name}_integration_guide.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"✅ Integration guide saved: {guide_path}")
        return str(guide_path)

def main():
    parser = argparse.ArgumentParser(description='Convert SLEAP models for iOS integration')
    parser.add_argument('--model', required=True, help='Path to SLEAP model (.h5)')
    parser.add_argument('--model-name', required=True, help='Model name for iOS integration')
    parser.add_argument('--output-dir', default='ios_models', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1
    
    # Initialize iOS integration
    ios_integration = iOSCoreMLIntegration(args.output_dir)
    
    try:
        print("🍎 Starting iOS Core ML integration...")
        
        # Convert to Core ML
        coreml_path = ios_integration.convert_sleap_to_coreml(args.model, args.model_name)
        
        if coreml_path:
            # Generate Swift integration code
            swift_path = ios_integration.generate_swift_integration_code(args.model_name, coreml_path)
            
            # Create integration guide
            guide_path = ios_integration.create_xcode_integration_guide(args.model_name, coreml_path, swift_path)
            
            print(f"""
🎉 iOS Integration Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📱 Core ML Model: {coreml_path}
📝 Swift Code: {swift_path}  
📋 Integration Guide: {guide_path}

Next Steps:
1. Add files to Xcode project
2. Follow integration guide
3. Test custom model vs Apple Vision
4. Setup A/B testing via Firebase
            """)
        else:
            print("❌ Core ML conversion failed")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ iOS integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)