#!/usr/bin/env python3
"""
YOLOv8 Pose Model Hailo8 Compilation Script for T3.2a
Compiles YOLOv8n-pose (human) and YOLOv8s-pose (dog) for Sprint 3
"""

import subprocess
import json
import sys
import requests
from pathlib import Path
from datetime import datetime

class YOLOPoseHailo8Compiler:
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.models_dir = self.repo_root / "pose_models"
        self.hailo_dir = self.repo_root / "hailo_compiled"
        self.models_dir.mkdir(exist_ok=True)
        self.hailo_dir.mkdir(exist_ok=True)
        
        # T3.2a Model Configuration
        self.models_config = {
            "yolov8n_pose_human": {
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt",
                "keypoints": 17,
                "species": "human",
                "target_fps": 45,
                "hw_arch": "hailo8"
            },
            "yolov8s_pose_dog": {
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt", 
                "keypoints": 20,  # Will need dog-specific training
                "species": "dog",
                "target_fps": 40,
                "hw_arch": "hailo8"
            }
        }
    
    def check_dependencies(self):
        """Check required dependencies"""
        print("🔍 Checking dependencies...")
        
        # Check Hailo SDK
        try:
            result = subprocess.run(['hailomz', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Hailo Model Zoo: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Hailo SDK not found")
            return False
            
        # Check Ultralytics
        try:
            import ultralytics
            print(f"✅ Ultralytics YOLO: {ultralytics.__version__}")
        except ImportError:
            print("⚠️  Ultralytics not found - will download pre-trained models")
            
        return True
    
    def download_model(self, model_name, config):
        """Download YOLO pose model"""
        model_path = self.models_dir / f"{model_name}.pt"
        
        if model_path.exists():
            print(f"✅ Model exists: {model_path}")
            return model_path
            
        print(f"📥 Downloading {model_name}...")
        try:
            response = requests.get(config["url"], stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"✅ Downloaded: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def convert_to_onnx(self, pt_path, model_name):
        """Convert PyTorch model to ONNX"""
        onnx_path = self.models_dir / f"{model_name}.onnx"
        
        if onnx_path.exists():
            print(f"✅ ONNX exists: {onnx_path}")
            return onnx_path
            
        print(f"🔄 Converting {model_name} to ONNX...")
        
        try:
            # Use ultralytics export if available
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            model.export(format='onnx', dynamic=False, imgsz=640)
            
            # Move to our directory structure
            exported_onnx = pt_path.with_suffix('.onnx')
            if exported_onnx.exists():
                exported_onnx.rename(onnx_path)
                print(f"✅ ONNX exported: {onnx_path}")
                return onnx_path
        except Exception as e:
            print(f"⚠️  Ultralytics export failed: {e}")
            
        # Fallback: create placeholder for manual conversion
        placeholder_content = {
            "model_type": "yolo_pose",
            "model_name": model_name,
            "status": "needs_manual_conversion",
            "pytorch_model": str(pt_path),
            "instructions": "Convert using: yolo export model=" + str(pt_path) + " format=onnx"
        }
        
        with open(onnx_path.with_suffix('.json'), 'w') as f:
            json.dump(placeholder_content, f, indent=2)
            
        print(f"📋 Manual conversion needed - see {onnx_path.with_suffix('.json')}")
        return None
    
    def compile_pose_model_hailo8(self, onnx_path, model_name, config):
        """Compile pose model for Hailo8"""
        if not onnx_path or not onnx_path.exists():
            print(f"❌ ONNX model not found: {onnx_path}")
            return None
            
        print(f"\n🎯 Compiling {model_name} for Hailo8 (T3.2a)")
        
        hef_path = self.hailo_dir / f"{model_name}_hailo8.hef"
        
        # Critical compilation - Fixed Hailo8 architecture
        cmd = [
            'hailomz', 'compile',
            str(onnx_path),
            '--hw-arch', 'hailo8',              # CRITICAL: hailo8 not hailo8l  
            '--performance',                     # Max performance
            '--name', f"{model_name}_hailo8",
            '--output-dir', str(self.hailo_dir),
            '--batch-size', '1',
            '--optimization', 'performance'
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {model_name} compilation successful!")
            print(f"📄 HEF: {hef_path}")
            return hef_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Compilation failed: {e}")
            # Store error for debugging
            error_log = self.hailo_dir / f"{model_name}_compile_error.log"
            with open(error_log, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\\n")
                f.write(f"stdout: {e.stdout}\\n")
                f.write(f"stderr: {e.stderr}\\n")
            print(f"📋 Error log: {error_log}")
            return None
    
    def run_t3_2a_compilation(self):
        """Execute T3.2a pose model compilation workflow"""
        print("🚀 T3.2a: Pose Model Compilation for Sprint 3")
        print("🎯 Target: YOLOv8n-pose (human) + YOLOv8s-pose (dog)")
        print("⚡ Architecture: Hailo8 (25% faster than Hailo8L)")
        
        if not self.check_dependencies():
            return False
            
        compiled_models = []
        
        for model_name, config in self.models_config.items():
            print(f"\n{'='*50}")
            print(f"Processing: {model_name}")
            print(f"Species: {config['species']}")
            print(f"Target FPS: {config['target_fps']}")
            
            # Download model
            pt_path = self.download_model(model_name, config)
            if not pt_path:
                continue
                
            # Convert to ONNX
            onnx_path = self.convert_to_onnx(pt_path, model_name)
            if not onnx_path:
                print(f"⚠️  {model_name}: Manual ONNX conversion required")
                continue
                
            # Compile for Hailo8
            hef_path = self.compile_pose_model_hailo8(onnx_path, model_name, config)
            if hef_path:
                compiled_models.append({
                    "model": model_name,
                    "hef_path": hef_path,
                    "config": config
                })
        
        # Generate T3.2a deployment summary
        self.generate_t3_2a_summary(compiled_models)
        
        return len(compiled_models) > 0
    
    def generate_t3_2a_summary(self, compiled_models):
        """Generate T3.2a completion summary"""
        summary = {
            "sprint_task": "T3.2a: Pose Model Compilation", 
            "priority": "P0 (Ready to Deploy)",
            "compilation_date": datetime.now().isoformat(),
            "architecture_fix": "Corrected hailo8l → hailo8 (+25% performance)",
            "compiled_models": len(compiled_models),
            "target_models": 3,  # TCN-VAE + 2 pose models
            "acceptance_criteria": "3 HEF files ready, >40 FPS each",
            "models": compiled_models,
            "next_steps": [
                "Test each model: hailortcli run model.hef --measure-fps",
                "Verify >40 FPS performance requirement",
                "Deploy to hailo_pipeline/artifacts/",
                "Continue to T3.3a: Multi-Model Scheduler"
            ]
        }
        
        summary_path = self.hailo_dir / "T3_2a_compilation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n{'='*60}")
        print(f"📋 T3.2a COMPILATION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Models compiled: {len(compiled_models)}/3")
        print(f"🎯 Architecture: Hailo8 (performance optimized)")
        print(f"📁 Summary: {summary_path}")
        
        for model in compiled_models:
            print(f"   • {model['model']}: {model['hef_path'].name}")
            
        if len(compiled_models) == 3:
            print(f"\n🎉 T3.2a COMPLETE! Ready for T3.3a Multi-Model Scheduler")
        else:
            print(f"\n⚠️  T3.2a PARTIAL: {3-len(compiled_models)} models need manual compilation")

if __name__ == "__main__":
    compiler = YOLOPoseHailo8Compiler()
    success = compiler.run_t3_2a_compilation()
    sys.exit(0 if success else 1)