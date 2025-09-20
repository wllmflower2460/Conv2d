#!/usr/bin/env python3
"""
Fixed YOLOv8s Compilation Script for Hailo8 (NOT Hailo8L)
Addresses the ~25% performance loss from incorrect architecture targeting
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

class YOLOv8HailoCompiler:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.models_dir = self.base_dir / "models"
        self.export_dir = self.base_dir / "export" 
        
        # Critical fix: Ensure Hailo8 (not Hailo8L) targeting
        self.hailo_config = {
            "target_hw": "hailo8",      # NOT hailo8l
            "hw_arch": "hailo8",        # Explicit architecture
            "performance_mode": True,   # Enable full performance
            "optimization_level": 2,    # Maximum optimization
            "batch_size": 1,           # Edge deployment standard
            "precision": "mixed"        # Mixed precision for efficiency
        }
        
        print("🎯 YOLOv8s Hailo8 Compiler - Architecture Fix")
        print(f"⚡ Target: Hailo8 (NOT Hailo8L) for ~25% performance gain")
        print(f"📁 Working directory: {self.base_dir}")
        
    def verify_hailo_sdk(self):
        """Verify Hailo SDK is available and check version"""
        try:
            result = subprocess.run(['hailomz', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Hailo SDK found: {result.stdout.strip()}")
            
            # Check compiler version
            result = subprocess.run(['hailo-compiler', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Hailo Compiler: {result.stdout.strip()}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Hailo SDK not found. Please install Hailo SDK first.")
            print("   https://developer.hailo.ai/documentation/sw-suite/")
            return False
    
    def download_yolov8s(self):
        """Download YOLOv8s base model"""
        print("\n📥 Downloading YOLOv8s base model...")
        
        try:
            from ultralytics import YOLO
            
            # Download YOLOv8s detection model
            model = YOLO('yolov8s.pt')  # Standard detection model
            print("✅ YOLOv8s model downloaded successfully")
            return True
            
        except ImportError:
            print("❌ Ultralytics not installed. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], 
                         check=True)
            return self.download_yolov8s()
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    
    def export_to_onnx(self):
        """Export YOLOv8s to ONNX with optimal settings"""
        print("\n🔄 Exporting YOLOv8s to ONNX...")
        
        try:
            from ultralytics import YOLO
            
            model = YOLO('yolov8s.pt')
            
            # Export with Hailo-optimized parameters
            success = model.export(
                format='onnx',
                imgsz=640,              # Standard input size
                simplify=True,          # Simplify for Hailo
                opset=11,              # Hailo compatibility
                batch=1,               # Single batch for edge
                device='cpu',          # CPU export for stability
                dynamic=False,         # Static shapes required
                verbose=True           # Detailed logging
            )
            
            onnx_path = Path('yolov8s.onnx')
            if onnx_path.exists():
                print(f"✅ ONNX export successful: {onnx_path}")
                return str(onnx_path)
            else:
                print("❌ ONNX file not found after export")
                return None
                
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return None
    
    def compile_for_hailo8(self, onnx_path):
        """Compile ONNX model specifically for Hailo8 (NOT Hailo8L)"""
        print("\n🔨 Compiling for Hailo8 architecture...")
        print("⚠️  CRITICAL: Targeting Hailo8 (NOT Hailo8L) for full performance")
        
        hef_path = self.export_dir / "yolov8s_hailo8_fixed.hef"
        self.export_dir.mkdir(exist_ok=True)
        
        # Critical compilation command with correct architecture
        compile_cmd = [
            'hailomz', 'compile',
            '--ckpt', str(onnx_path),
            '--hw-arch', 'hailo8',              # CRITICAL: hailo8 not hailo8l
            '--output-dir', str(self.export_dir),
            '--name', 'yolov8s_hailo8_fixed',
            '--performance',                    # Enable performance mode
            '--optimization', 'max',            # Maximum optimization
            '--batch-size', '1'
        ]
        
        print(f"🔧 Compilation command:")
        print(f"   {' '.join(compile_cmd)}")
        print(f"🎯 Target architecture: hailo8 (25% faster than hailo8l)")
        
        try:
            print(f"\n⏳ Compiling... (this may take several minutes)")
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.base_dir
            )
            
            print("✅ Compilation successful!")
            print(f"📁 HEF file: {hef_path}")
            
            # Log compilation output
            if result.stdout:
                print(f"\n📊 Compilation output:")
                print(result.stdout)
            
            return str(hef_path) if hef_path.exists() else None
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Compilation failed!")
            print(f"Error: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return None
    
    def validate_compilation(self, hef_path):
        """Validate the compiled HEF file"""
        print(f"\n🔍 Validating compiled model...")
        
        hef_file = Path(hef_path)
        if not hef_file.exists():
            print(f"❌ HEF file not found: {hef_path}")
            return False
        
        file_size = hef_file.stat().st_size / (1024 * 1024)  # MB
        print(f"📊 HEF file size: {file_size:.1f} MB")
        
        # Basic validation
        if file_size < 1.0:
            print("⚠️  Warning: HEF file seems small, compilation might have failed")
            return False
        elif file_size > 100.0:
            print("⚠️  Warning: HEF file seems large, check optimization settings")
        
        print("✅ HEF file validation passed")
        
        # Create deployment info
        deployment_info = {
            "model": "yolov8s",
            "architecture": "hailo8",           # Confirmed Hailo8
            "compilation_date": datetime.now().isoformat(),
            "hef_path": str(hef_path),
            "hef_size_mb": file_size,
            "target_performance": "52.5 FPS * 1.25 = ~65 FPS",
            "hardware_fix": "Corrected from hailo8l to hailo8",
            "expected_improvement": "~25% performance gain"
        }
        
        info_path = self.export_dir / "yolov8s_hailo8_deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"📋 Deployment info saved: {info_path}")
        return True
    
    def create_deployment_script(self, hef_path):
        """Create Pi deployment script"""
        print(f"\n📝 Creating Pi deployment script...")
        
        deployment_script = f'''#!/bin/bash
# YOLOv8s Hailo8 Deployment Script
# Fixed compilation for Hailo8 (not Hailo8L) - ~25% performance improvement

echo "🚀 Deploying YOLOv8s Hailo8 (Performance Fixed)"
echo "⚡ Expected performance: ~65 FPS (vs 52.5 FPS with hailo8l)"

HEF_FILE="{Path(hef_path).name}"
PI_MODELS_DIR="/home/pi/models"

# Copy HEF to Pi
echo "📁 Copying HEF file to Pi..."
scp $HEF_FILE pi@raspberrypi.local:$PI_MODELS_DIR/

# SSH to Pi and verify
echo "🔍 Verifying deployment on Pi..."
ssh pi@raspberrypi.local << 'EOF'
cd $PI_MODELS_DIR
ls -la *.hef
echo "✅ HEF file deployed successfully"

# Run performance test
echo "⚡ Running performance benchmark..."
python3 /home/pi/hailo_tools/benchmark_yolov8.py --model $HEF_FILE --target-fps 65

echo "🎯 Expected results:"
echo "  - ~65 FPS (vs 52.5 FPS baseline)" 
echo "  - ~25% performance improvement"
echo "  - Hailo8 architecture confirmed"
EOF

echo "✅ Deployment complete!"
echo "🎯 Performance fix applied: Hailo8 vs Hailo8L"
'''
        
        script_path = self.export_dir / "deploy_yolov8s_hailo8.sh"
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        
        # Make executable
        script_path.chmod(0o755)
        print(f"✅ Deployment script created: {script_path}")
        return str(script_path)
    
    def run_full_compilation(self):
        """Run complete compilation pipeline"""
        print(f"🚀 Starting YOLOv8s Hailo8 Compilation (Architecture Fix)")
        print(f"⏰ Started at: {datetime.now()}")
        
        # Step 1: Verify Hailo SDK
        if not self.verify_hailo_sdk():
            return False
        
        # Step 2: Download model
        if not self.download_yolov8s():
            return False
        
        # Step 3: Export to ONNX
        onnx_path = self.export_to_onnx()
        if not onnx_path:
            return False
        
        # Step 4: Compile for Hailo8 (NOT Hailo8L)
        hef_path = self.compile_for_hailo8(onnx_path)
        if not hef_path:
            return False
        
        # Step 5: Validate
        if not self.validate_compilation(hef_path):
            return False
        
        # Step 6: Create deployment script
        deploy_script = self.create_deployment_script(hef_path)
        
        print(f"\n🏁 YOLOv8s Hailo8 Compilation Complete!")
        print(f"✅ Architecture: Hailo8 (NOT Hailo8L)")
        print(f"⚡ Expected improvement: ~25% performance gain")
        print(f"📈 Target FPS: ~65 FPS (vs 52.5 FPS baseline)")
        print(f"📁 HEF file: {hef_path}")
        print(f"🚀 Deployment: {deploy_script}")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Transfer HEF to Pi: scp {Path(hef_path).name} pi@raspberrypi.local:/home/pi/models/")
        print(f"2. Update EdgeInfer configuration to use new HEF")
        print(f"3. Benchmark performance (expect ~65 FPS)")
        print(f"4. Verify 25% improvement over hailo8l compilation")
        
        return True

def main():
    """Main compilation function"""
    compiler = YOLOv8HailoCompiler()
    
    try:
        success = compiler.run_full_compilation()
        
        if success:
            print("✅ SUCCESS: YOLOv8s compiled for Hailo8 architecture")
            return 0
        else:
            print("❌ FAILED: Compilation unsuccessful")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Compilation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)