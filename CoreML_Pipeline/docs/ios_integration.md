# iOS Integration Guide for Dog Behavior Classifier

## Table of Contents
1. [Model Integration](#model-integration)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Keypoint Detection](#keypoint-detection)
4. [Temporal Buffering](#temporal-buffering)
5. [Real-time Inference](#real-time-inference)
6. [Performance Optimization](#performance-optimization)
7. [Testing & Debugging](#testing--debugging)

## Model Integration

### Adding Model to Xcode

1. **Import the Model**
   - Drag `DogBehaviorClassifier.mlpackage` into your Xcode project navigator
   - Select "Copy items if needed"
   - Add to your app target

2. **Generated Interface**
   Xcode automatically generates a Swift interface:
   ```swift
   class DogBehaviorClassifier {
       func prediction(pose_sequence: MLMultiArray) throws -> DogBehaviorClassifierOutput
   }
   ```

### Model Configuration

```swift
import CoreML

// Configure model with compute units
let config = MLModelConfiguration()
config.computeUnits = .all  // Use Neural Engine when available

// Load model
guard let model = try? DogBehaviorClassifier(configuration: config) else {
    fatalError("Failed to load Dog Behavior Classifier model")
}
```

## Data Flow Architecture

```
Camera/Video → Pose Detection → Keypoint Buffer → Model Inference → Behavior Output
```

### Complete Pipeline Example

```swift
class DogBehaviorAnalyzer {
    private let model: DogBehaviorClassifier
    private var keypointBuffer: KeypointBuffer
    private let bufferSize = 100  // 100 frames
    
    init() throws {
        self.model = try DogBehaviorClassifier(configuration: MLModelConfiguration())
        self.keypointBuffer = KeypointBuffer(size: bufferSize)
    }
    
    func process(keypoints: [CGPoint], imageSize: CGSize) -> String? {
        // Normalize keypoints
        let normalized = normalizeKeypoints(keypoints, imageSize: imageSize)
        
        // Add to buffer
        keypointBuffer.add(normalized)
        
        // Check if buffer is ready
        guard keypointBuffer.isFull else { return nil }
        
        // Prepare input
        let input = prepareModelInput(from: keypointBuffer)
        
        // Run inference
        return predict(with: input)
    }
}
```

## Keypoint Detection

### Using Vision Framework for Pose Detection

```swift
import Vision

class PoseDetector {
    private let animalBodyPoseRequest = VNDetectAnimalBodyPoseRequest()
    
    func detectPose(in image: CVPixelBuffer, completion: @escaping ([CGPoint]?) -> Void) {
        let handler = VNImageRequestHandler(cvPixelBuffer: image, options: [:])
        
        do {
            try handler.perform([animalBodyPoseRequest])
            
            guard let observations = animalBodyPoseRequest.results?.first else {
                completion(nil)
                return
            }
            
            let keypoints = extractDogKeypoints(from: observations)
            completion(keypoints)
            
        } catch {
            print("Pose detection failed: \(error)")
            completion(nil)
        }
    }
    
    private func extractDogKeypoints(from observation: VNAnimalBodyPoseObservation) -> [CGPoint] {
        // Map Vision points to our 24-point structure
        var keypoints: [CGPoint] = []
        
        // Define mapping from Vision to our structure
        let keypointNames: [VNAnimalBodyPoseObservation.JointName] = [
            .snout,           // nose
            .leftEye,         // left_eye
            .rightEye,        // right_eye
            .leftEar,         // left_ear
            .rightEar,        // right_ear
            // ... map all 24 points
        ]
        
        for jointName in keypointNames {
            if let point = try? observation.recognizedPoint(jointName) {
                keypoints.append(CGPoint(x: point.x, y: point.y))
            } else {
                keypoints.append(CGPoint.zero)  // Missing point
            }
        }
        
        return keypoints
    }
}
```

## Temporal Buffering

### Keypoint Buffer Implementation

```swift
class KeypointBuffer {
    private var buffer: [[Float]] = []
    private let maxSize: Int
    
    init(size: Int) {
        self.maxSize = size
    }
    
    var isFull: Bool {
        return buffer.count >= maxSize
    }
    
    func add(_ keypoints: [Float]) {
        buffer.append(keypoints)
        
        // Sliding window: remove oldest if full
        if buffer.count > maxSize {
            buffer.removeFirst()
        }
    }
    
    func getMLArray() -> MLMultiArray? {
        guard isFull else { return nil }
        
        // Create MLMultiArray with shape [1, 48, 100]
        guard let array = try? MLMultiArray(shape: [1, 48, NSNumber(value: maxSize)], 
                                           dataType: .float32) else {
            return nil
        }
        
        // Fill array with buffer data
        for frameIdx in 0..<maxSize {
            for keypointIdx in 0..<48 {
                let index = [0, keypointIdx, frameIdx] as [NSNumber]
                array[index] = NSNumber(value: buffer[frameIdx][keypointIdx])
            }
        }
        
        return array
    }
    
    func reset() {
        buffer.removeAll()
    }
}
```

## Real-time Inference

### Complete Inference Pipeline

```swift
class RealTimeDogBehaviorClassifier {
    private let model: DogBehaviorClassifier
    private let keypointBuffer = KeypointBuffer(size: 100)
    private var lastPrediction: String?
    private var predictionConfidence: Float = 0.0
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try DogBehaviorClassifier(configuration: config)
    }
    
    func processFrame(keypoints: [CGPoint], imageSize: CGSize) -> BehaviorPrediction? {
        // 1. Normalize keypoints to [0,1] range
        let normalized = keypoints.map { point in
            return [
                Float(point.x / imageSize.width),
                Float(point.y / imageSize.height)
            ]
        }.flatMap { $0 }  // Flatten to [x0,y0,x1,y1,...]
        
        // 2. Add to temporal buffer
        keypointBuffer.add(normalized)
        
        // 3. Check if ready for inference
        guard keypointBuffer.isFull,
              let inputArray = keypointBuffer.getMLArray() else {
            return nil
        }
        
        // 4. Run inference
        do {
            let output = try model.prediction(pose_sequence: inputArray)
            
            // 5. Process output
            let behavior = processPrediction(output)
            return behavior
            
        } catch {
            print("Inference failed: \(error)")
            return nil
        }
    }
    
    private func processPrediction(_ output: DogBehaviorClassifierOutput) -> BehaviorPrediction {
        // Extract probabilities
        let probs = output.behavior_probs
        
        // Find max probability
        var maxProb: Float = 0.0
        var maxIndex = 0
        
        for i in 0..<21 {
            let prob = probs[[NSNumber(value: i)]].floatValue
            if prob > maxProb {
                maxProb = prob
                maxIndex = i
            }
        }
        
        let behaviors = ["sit", "down", "stand", "stay", "lying", 
                        "heel", "come", "fetch", "drop", "wait",
                        "leave_it", "walking", "trotting", "running", 
                        "jumping", "spinning", "rolling", "playing",
                        "alert", "sniffing", "looking"]
        
        return BehaviorPrediction(
            behavior: behaviors[maxIndex],
            confidence: maxProb,
            allProbabilities: extractAllProbabilities(from: probs)
        )
    }
}

struct BehaviorPrediction {
    let behavior: String
    let confidence: Float
    let allProbabilities: [String: Float]
}
```

## Performance Optimization

### 1. Frame Rate Management

```swift
class FrameRateManager {
    private var lastProcessTime = Date()
    private let targetFPS: Double = 10  // Process at 10 FPS instead of 30
    
    var shouldProcessFrame: Bool {
        let elapsed = Date().timeIntervalSince(lastProcessTime)
        if elapsed >= 1.0 / targetFPS {
            lastProcessTime = Date()
            return true
        }
        return false
    }
}
```

### 2. Async Processing

```swift
class AsyncBehaviorClassifier {
    private let processingQueue = DispatchQueue(label: "behavior.inference", qos: .userInitiated)
    private let model: DogBehaviorClassifier
    
    func processAsync(keypoints: [CGPoint], 
                     imageSize: CGSize,
                     completion: @escaping (BehaviorPrediction?) -> Void) {
        processingQueue.async { [weak self] in
            let result = self?.processFrame(keypoints: keypoints, imageSize: imageSize)
            
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }
}
```

### 3. Memory Management

```swift
class EfficientBuffer {
    private var circularBuffer: UnsafeMutablePointer<Float>
    private let capacity: Int
    private var writeIndex = 0
    
    init(frames: Int, keypointsPerFrame: Int) {
        self.capacity = frames * keypointsPerFrame
        self.circularBuffer = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
    }
    
    deinit {
        circularBuffer.deallocate()
    }
    
    func write(_ data: [Float]) {
        data.enumerated().forEach { idx, value in
            let bufferIdx = (writeIndex + idx) % capacity
            circularBuffer[bufferIdx] = value
        }
        writeIndex = (writeIndex + data.count) % capacity
    }
}
```

## Testing & Debugging

### Unit Test Example

```swift
import XCTest
@testable import YourApp

class DogBehaviorClassifierTests: XCTestCase {
    var classifier: RealTimeDogBehaviorClassifier!
    
    override func setUp() {
        super.setUp()
        classifier = try! RealTimeDogBehaviorClassifier()
    }
    
    func testKeypointNormalization() {
        let keypoints = [CGPoint(x: 100, y: 200)]
        let imageSize = CGSize(width: 200, height: 400)
        
        let normalized = normalizeKeypoints(keypoints, imageSize: imageSize)
        
        XCTAssertEqual(normalized[0], 0.5, accuracy: 0.001)
        XCTAssertEqual(normalized[1], 0.5, accuracy: 0.001)
    }
    
    func testBufferFilling() {
        let buffer = KeypointBuffer(size: 100)
        
        // Add 100 frames
        for _ in 0..<100 {
            buffer.add(Array(repeating: 0.5, count: 48))
        }
        
        XCTAssertTrue(buffer.isFull)
        XCTAssertNotNil(buffer.getMLArray())
    }
    
    func testModelInference() {
        // Create synthetic input
        let input = try! MLMultiArray(shape: [1, 48, 100], dataType: .float32)
        
        // Fill with test data
        for i in 0..<48*100 {
            input[i] = NSNumber(value: Float.random(in: 0...1))
        }
        
        // Run inference
        let output = try! classifier.model.prediction(pose_sequence: input)
        
        XCTAssertNotNil(output.behavior_probs)
    }
}
```

### Debug Visualization

```swift
extension UIView {
    func drawKeypoints(_ keypoints: [CGPoint], 
                      connections: [(Int, Int)] = []) {
        layer.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        // Draw keypoints
        keypoints.enumerated().forEach { idx, point in
            let dot = CAShapeLayer()
            dot.path = UIBezierPath(ovalIn: CGRect(x: point.x - 3, 
                                                   y: point.y - 3, 
                                                   width: 6, 
                                                   height: 6)).cgPath
            dot.fillColor = UIColor.red.cgColor
            layer.addSublayer(dot)
        }
        
        // Draw skeleton connections
        for (from, to) in connections {
            let line = CAShapeLayer()
            let path = UIBezierPath()
            path.move(to: keypoints[from])
            path.addLine(to: keypoints[to])
            line.path = path.cgPath
            line.strokeColor = UIColor.blue.cgColor
            line.lineWidth = 2
            layer.addSublayer(line)
        }
    }
}
```

## Common Issues & Solutions

### Issue 1: Model Not Loading
```swift
// Check model file exists in bundle
guard let modelURL = Bundle.main.url(forResource: "DogBehaviorClassifier", 
                                     withExtension: "mlmodelc") else {
    print("Model file not found in bundle")
    return
}
```

### Issue 2: Memory Warnings
```swift
// Use autorelease pool for batch processing
autoreleasepool {
    for frame in frames {
        processFrame(frame)
    }
}
```

### Issue 3: Low Confidence Predictions
```swift
// Apply confidence threshold
if prediction.confidence < 0.7 {
    return "uncertain"
}
```

## SwiftUI Integration

```swift
import SwiftUI
import Combine

class BehaviorViewModel: ObservableObject {
    @Published var currentBehavior: String = "unknown"
    @Published var confidence: Float = 0.0
    
    private var classifier: RealTimeDogBehaviorClassifier?
    
    init() {
        do {
            classifier = try RealTimeDogBehaviorClassifier()
        } catch {
            print("Failed to initialize classifier: \(error)")
        }
    }
    
    func updatePose(_ keypoints: [CGPoint], imageSize: CGSize) {
        guard let prediction = classifier?.processFrame(keypoints: keypoints, 
                                                       imageSize: imageSize) else {
            return
        }
        
        DispatchQueue.main.async {
            self.currentBehavior = prediction.behavior
            self.confidence = prediction.confidence
        }
    }
}

struct BehaviorView: View {
    @StateObject private var viewModel = BehaviorViewModel()
    
    var body: some View {
        VStack {
            Text("Behavior: \(viewModel.currentBehavior)")
                .font(.title)
            
            ProgressView(value: viewModel.confidence)
                .progressViewStyle(.linear)
            
            Text("Confidence: \(Int(viewModel.confidence * 100))%")
                .font(.caption)
        }
        .padding()
    }
}
```

---

## Next Steps

1. Implement pose detection using Vision framework
2. Set up temporal buffering for 100-frame windows
3. Integrate model inference pipeline
4. Add UI for displaying predictions
5. Optimize for real-time performance

For complete example code, see `ios_examples/DogBehaviorPredictor.swift`