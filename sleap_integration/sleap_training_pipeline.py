#!/usr/bin/env python3
"""
SLEAP Training Pipeline for DataDogs Animal Pose Estimation
Complete pipeline for training, evaluation, and deployment of custom animal pose models

Usage:
  python sleap_training_pipeline.py --dataset dataset.slp --output-dir models/
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil

try:
    import sleap
    import tensorflow as tf
    import numpy as np
    import cv2
    from sleap import Labels, Video, LabeledFrame
    from sleap.nn.training import Trainer
    from sleap.nn.config import TrainingJobConfig
except ImportError:
    print("❌ SLEAP not installed. Install with:")
    print("conda install sleap -c sleap -c nvidia -c conda-forge")
    sys.exit(1)

class SLEAPTrainingPipeline:
    """Complete pipeline for SLEAP model training and deployment"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.evaluation_dir = self.output_dir / "evaluation"
        self.exports_dir = self.output_dir / "exports"
        
        # Create subdirectories
        for dir_path in [self.models_dir, self.logs_dir, self.evaluation_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def prepare_dataset(self, dataset_path: str, validation_split: float = 0.2) -> Tuple[str, str]:
        """Prepare training and validation datasets"""
        print(f"📊 Preparing dataset: {dataset_path}")
        
        # Load dataset
        labels = sleap.load_file(dataset_path)
        
        print(f"📋 Dataset statistics:")
        print(f"   • Total frames: {len(labels.labeled_frames)}")
        print(f"   • Total instances: {len(labels.instances)}")
        print(f"   • Videos: {len(labels.videos)}")
        print(f"   • Skeleton nodes: {len(labels.skeletons[0].nodes) if labels.skeletons else 0}")
        
        # Split dataset
        n_total = len(labels.labeled_frames)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        
        # Random split (you might want to do video-based split for better generalization)
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create training dataset
        train_frames = [labels.labeled_frames[i] for i in train_indices]
        train_labels = sleap.Labels(train_frames)
        train_path = str(self.output_dir / "train_dataset.slp")
        train_labels.save(train_path)
        
        # Create validation dataset
        val_frames = [labels.labeled_frames[i] for i in val_indices]
        val_labels = sleap.Labels(val_frames)
        val_path = str(self.output_dir / "val_dataset.slp")
        val_labels.save(val_path)
        
        print(f"✅ Dataset split complete:")
        print(f"   • Training: {len(train_frames)} frames ({train_path})")
        print(f"   • Validation: {len(val_frames)} frames ({val_path})")
        
        return train_path, val_path
    
    def create_training_configs(self, train_dataset: str, val_dataset: str) -> List[str]:
        """Create multiple training configurations for different model architectures"""
        print("📝 Creating training configurations...")
        
        configs = []
        
        # Configuration 1: Centered Instance Model (Recommended for single animal)
        config1 = {
            "data": {
                "labels": {
                    "training_labels": train_dataset,
                    "validation_labels": val_dataset,
                    "split_by_video": False
                },
                "preprocessing": {
                    "ensure_rgb": True,
                    "ensure_grayscale": False,
                    "input_scaling": 1.0,
                    "pad_to_stride": 16,
                    "resize_and_pad_to_target": True,
                    "target_height": 512,
                    "target_width": 512
                },
                "instance_cropping": {
                    "center_on_part": "spine_mid",
                    "crop_size": 384,
                    "crop_size_detection_padding": 16
                }
            },
            "model": {
                "backbone": {
                    "unet": {
                        "stem_stride": None,
                        "max_stride": 16,
                        "output_stride": 2,
                        "filters": 64,
                        "filters_rate": 2,
                        "middle_block": True,
                        "up_interpolate": True,
                        "stacks": 1
                    }
                },
                "heads": {
                    "centered_instance": {
                        "anchor_part": "spine_mid",
                        "part_names": None,
                        "sigma": 2.5,
                        "output_stride": 2,
                        "loss_weight": 1.0,
                        "offset_refinement": True
                    }
                }
            },
            "trainer": {
                "training_labels": train_dataset,
                "validation_labels": val_dataset,
                "tensorboard": {"write_logs": True, "log_dir": str(self.logs_dir / "centered_instance")},
                "validation_frequency": 5,
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "plateau_min_delta": 1e-08,
                    "plateau_patience": 15,
                    "plateau_reduce_lr_on_plateau": True,
                    "plateau_reduce_lr_factor": 0.5,
                    "plateau_reduce_lr_patience": 8
                },
                "learning_rate": 1e-4,
                "epochs": 200,
                "batch_size": 8,
                "optimizer": "adam",
                "save_viz": True,
                "save_viz_frequency": 20,
                "seed": 1000
            }
        }
        
        # Configuration 2: Multi-Instance Model (For multiple animals)
        config2 = {
            "data": {
                "labels": {
                    "training_labels": train_dataset,
                    "validation_labels": val_dataset,
                    "split_by_video": False
                },
                "preprocessing": {
                    "ensure_rgb": True,
                    "input_scaling": 1.0,
                    "pad_to_stride": 16,
                    "resize_and_pad_to_target": True,
                    "target_height": 640,
                    "target_width": 640
                }
            },
            "model": {
                "backbone": {
                    "unet": {
                        "max_stride": 16,
                        "output_stride": 4,
                        "filters": 64,
                        "filters_rate": 2,
                        "stacks": 1
                    }
                },
                "heads": {
                    "multi_instance": {
                        "confmaps": {
                            "part_names": None,
                            "sigma": 2.5,
                            "output_stride": 4,
                            "loss_weight": 1.0
                        },
                        "pafs": {
                            "edges": None,
                            "sigma": 75,
                            "output_stride": 4,
                            "loss_weight": 1.0
                        }
                    }
                }
            },
            "trainer": {
                "training_labels": train_dataset,
                "validation_labels": val_dataset,
                "tensorboard": {"write_logs": True, "log_dir": str(self.logs_dir / "multi_instance")},
                "validation_frequency": 5,
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "plateau_patience": 20
                },
                "learning_rate": 1e-4,
                "epochs": 300,
                "batch_size": 4,
                "optimizer": "adam",
                "save_viz": True,
                "seed": 1000
            }
        }
        
        # Configuration 3: High-Resolution Model (Maximum accuracy)
        config3 = {
            "data": {
                "labels": {
                    "training_labels": train_dataset,
                    "validation_labels": val_dataset,
                    "split_by_video": False
                },
                "preprocessing": {
                    "ensure_rgb": True,
                    "input_scaling": 1.0,
                    "pad_to_stride": 32,
                    "resize_and_pad_to_target": True,
                    "target_height": 768,
                    "target_width": 768
                },
                "instance_cropping": {
                    "center_on_part": "spine_mid",
                    "crop_size": 512,
                    "crop_size_detection_padding": 32
                }
            },
            "model": {
                "backbone": {
                    "unet": {
                        "max_stride": 32,
                        "output_stride": 2,
                        "filters": 96,
                        "filters_rate": 2,
                        "stacks": 2,
                        "middle_block": True
                    }
                },
                "heads": {
                    "centered_instance": {
                        "anchor_part": "spine_mid",
                        "sigma": 1.5,
                        "output_stride": 2,
                        "loss_weight": 1.0,
                        "offset_refinement": True
                    }
                }
            },
            "trainer": {
                "training_labels": train_dataset,
                "validation_labels": val_dataset,
                "tensorboard": {"write_logs": True, "log_dir": str(self.logs_dir / "high_resolution")},
                "validation_frequency": 10,
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "plateau_patience": 25
                },
                "learning_rate": 5e-5,
                "epochs": 400,
                "batch_size": 2,
                "optimizer": "adam",
                "save_viz": True,
                "seed": 1000
            }
        }
        
        # Save configurations
        config_names = ["centered_instance", "multi_instance", "high_resolution"]
        config_data = [config1, config2, config3]
        
        for name, config in zip(config_names, config_data):
            config_path = self.output_dir / f"{name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            configs.append(str(config_path))
            print(f"✅ Config saved: {config_path}")
        
        return configs
    
    def train_model(self, config_path: str, model_name: str) -> Optional[str]:
        """Train a SLEAP model using the specified configuration"""
        print(f"🚀 Starting training: {model_name}")
        
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Run SLEAP training
            cmd = [
                "sleap-train",
                config_path,
                "--run-name", model_name,
                "--save-dir", str(model_dir)
            ]
            
            print(f"📝 Training command: {' '.join(cmd)}")
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.output_dir)
            )
            
            if result.returncode == 0:
                print(f"✅ Training completed: {model_name}")
                
                # Find the trained model
                trained_models = list(model_dir.glob("*.h5"))
                if trained_models:
                    return str(trained_models[0])
                else:
                    print(f"⚠️  No .h5 model found in {model_dir}")
                    return None
            else:
                print(f"❌ Training failed: {model_name}")
                print(f"Error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Training error for {model_name}: {e}")
            return None
    
    def evaluate_model(self, model_path: str, test_dataset: str, model_name: str) -> Dict:
        """Evaluate trained model performance"""
        print(f"📊 Evaluating model: {model_name}")
        
        eval_dir = self.evaluation_dir / model_name
        eval_dir.mkdir(exist_ok=True)
        
        try:
            # Run SLEAP evaluation
            cmd = [
                "sleap-track",
                test_dataset,
                "--models", model_path,
                "--output", str(eval_dir / "predictions.slp"),
                "--verbosity", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load predictions and ground truth for evaluation
                predictions = sleap.load_file(str(eval_dir / "predictions.slp"))
                ground_truth = sleap.load_file(test_dataset)
                
                # Calculate metrics
                metrics = self.calculate_metrics(predictions, ground_truth)
                
                # Save evaluation results
                eval_results = {
                    "model_name": model_name,
                    "model_path": model_path,
                    "test_dataset": test_dataset,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                }
                
                with open(eval_dir / "evaluation_results.json", 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                print(f"✅ Evaluation completed: {model_name}")
                print(f"📈 OKS@0.5: {metrics.get('oks_50', 'N/A'):.3f}")
                print(f"📈 PCK@0.1: {metrics.get('pck_10', 'N/A'):.3f}")
                
                return eval_results
            else:
                print(f"❌ Evaluation failed: {result.stderr}")
                return {}
                
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {}
    
    def calculate_metrics(self, predictions: Labels, ground_truth: Labels) -> Dict:
        """Calculate pose estimation metrics"""
        # Simplified metrics calculation
        # In practice, you'd want more sophisticated evaluation
        
        metrics = {
            "total_predictions": len(predictions.labeled_frames),
            "total_ground_truth": len(ground_truth.labeled_frames),
            "mean_confidence": 0.0,
            "detection_rate": 0.0,
            "oks_50": 0.0,  # Object Keypoint Similarity at 0.5 threshold
            "pck_10": 0.0   # Percentage of Correct Keypoints at 0.1 threshold
        }
        
        # Add your metric calculations here
        # This is a placeholder implementation
        
        return metrics
    
    def export_for_ios(self, model_path: str, model_name: str) -> Optional[str]:
        """Export model for iOS Core ML deployment"""
        print(f"📱 Exporting for iOS: {model_name}")
        
        export_dir = self.exports_dir / model_name
        export_dir.mkdir(exist_ok=True)
        
        try:
            # Export to TensorFlow Lite first
            tflite_path = export_dir / f"{model_name}.tflite"
            
            cmd = [
                "sleap-export",
                model_path,
                "--format", "tflite",
                "--output", str(tflite_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ TensorFlow Lite export: {tflite_path}")
                
                # Convert to Core ML (requires additional tools)
                coreml_path = export_dir / f"{model_name}.mlmodel"
                
                try:
                    # This requires coremltools
                    import coremltools as ct
                    
                    # Load TFLite model and convert
                    model = ct.convert(
                        str(tflite_path),
                        inputs=[ct.ImageType(name="input", shape=(1, 384, 384, 3))],
                        outputs=[ct.TensorType(name="output")]
                    )
                    
                    # Add metadata
                    model.short_description = f"DataDogs Animal Pose Estimation - {model_name}"
                    model.author = "DataDogs Platform"
                    model.version = "1.0"
                    
                    model.save(str(coreml_path))
                    print(f"✅ Core ML export: {coreml_path}")
                    
                    return str(coreml_path)
                    
                except ImportError:
                    print("⚠️  coremltools not available. Install with: pip install coremltools")
                    return str(tflite_path)
                    
            else:
                print(f"❌ Export failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Export error: {e}")
            return None
    
    def create_deployment_package(self, best_model_path: str, model_name: str) -> str:
        """Create deployment package with model and metadata"""
        print(f"📦 Creating deployment package: {model_name}")
        
        package_dir = self.exports_dir / f"{model_name}_deployment"
        package_dir.mkdir(exist_ok=True)
        
        # Copy model files
        if best_model_path:
            shutil.copy2(best_model_path, package_dir)
        
        # Create deployment metadata
        deployment_info = {
            "model_name": model_name,
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "framework": "SLEAP",
            "target_platform": "iOS",
            "input_shape": [384, 384, 3],
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "neck", "left_front_shoulder", "left_front_elbow", "left_front_wrist", "left_front_paw",
                "right_front_shoulder", "right_front_elbow", "right_front_wrist", "right_front_paw",
                "spine_mid", "left_back_shoulder", "left_back_elbow", "left_back_wrist", "left_back_paw",
                "right_back_shoulder", "right_back_elbow", "right_back_wrist", "right_back_paw",
                "tail_base", "tail_mid", "tail_tip"
            ],
            "usage_instructions": {
                "preprocessing": "Resize to 384x384, normalize to [0,1]",
                "postprocessing": "Apply confidence threshold, NMS for multiple instances",
                "integration": "Use with DataDogs Firebase distributed learning pipeline"
            }
        }
        
        with open(package_dir / "deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Create README
        readme_content = f"""# {model_name} Deployment Package

## Model Information
- **Name**: {model_name}
- **Framework**: SLEAP
- **Target**: iOS Core ML
- **Input Size**: 384x384x3
- **Keypoints**: 25 animal pose landmarks

## Integration with DataDogs

1. **iOS Integration**:
   ```swift
   // Load Core ML model
   let model = try {model_name}().model
   
   // Process frame
   let prediction = try model.prediction(from: pixelBuffer)
   ```

2. **Firebase Pipeline**:
   - Uncertain predictions → Firebase Storage
   - CVAT annotation → Improved labels
   - Model retraining → Performance improvement

## File Structure
- `{model_name}.mlmodel` - Core ML model for iOS
- `deployment_info.json` - Model metadata
- `README.md` - This file

## Next Steps
1. Import model into Xcode project
2. Update PoseEstimationViewModel to use custom model
3. Configure uncertainty detection thresholds
4. Enable A/B testing via Firebase RemoteConfig
"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Deployment package created: {package_dir}")
        return str(package_dir)

def main():
    parser = argparse.ArgumentParser(description='SLEAP Training Pipeline for DataDogs')
    parser.add_argument('--dataset', required=True, help='Path to SLEAP dataset (.slp)')
    parser.add_argument('--output-dir', default='sleap_training_output', help='Output directory')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--train-all', action='store_true', help='Train all model configurations')
    parser.add_argument('--model-config', help='Specific config to train (centered_instance, multi_instance, high_resolution)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
        return 1
    
    # Initialize pipeline
    pipeline = SLEAPTrainingPipeline(args.output_dir)
    
    try:
        print("🚀 Starting SLEAP Training Pipeline for DataDogs")
        print("=" * 50)
        
        # Prepare datasets
        train_dataset, val_dataset = pipeline.prepare_dataset(args.dataset, args.validation_split)
        
        # Create training configurations
        configs = pipeline.create_training_configs(train_dataset, val_dataset)
        
        # Determine which models to train
        if args.model_config:
            # Train specific model
            config_map = {
                "centered_instance": configs[0],
                "multi_instance": configs[1], 
                "high_resolution": configs[2]
            }
            
            if args.model_config in config_map:
                config_path = config_map[args.model_config]
                model_path = pipeline.train_model(config_path, args.model_config)
                
                if model_path:
                    # Evaluate model
                    eval_results = pipeline.evaluate_model(model_path, val_dataset, args.model_config)
                    
                    # Export for iOS
                    exported_model = pipeline.export_for_ios(model_path, args.model_config)
                    
                    # Create deployment package
                    if exported_model:
                        package_path = pipeline.create_deployment_package(exported_model, args.model_config)
                        print(f"🎉 Ready for deployment: {package_path}")
            else:
                print(f"❌ Unknown model config: {args.model_config}")
                return 1
                
        elif args.train_all:
            # Train all configurations
            model_names = ["centered_instance", "multi_instance", "high_resolution"]
            results = []
            
            for config_path, model_name in zip(configs, model_names):
                print(f"\n🔄 Training {model_name}...")
                
                model_path = pipeline.train_model(config_path, model_name)
                if model_path:
                    eval_results = pipeline.evaluate_model(model_path, val_dataset, model_name)
                    results.append((model_name, model_path, eval_results))
            
            # Select best model
            if results:
                best_model = max(results, key=lambda x: x[2].get('metrics', {}).get('oks_50', 0))
                best_name, best_path, best_eval = best_model
                
                print(f"\n🏆 Best model: {best_name}")
                
                # Export best model
                exported_model = pipeline.export_for_ios(best_path, best_name)
                if exported_model:
                    package_path = pipeline.create_deployment_package(exported_model, best_name)
                    print(f"🎉 Ready for deployment: {package_path}")
        else:
            print("❌ Specify --model-config or --train-all")
            return 1
        
        print("\n✅ SLEAP training pipeline completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)