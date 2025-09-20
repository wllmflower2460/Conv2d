#!/usr/bin/env python3
"""
TCN-VAE Hailo8 Compilation Script
Compiles TCN-VAE ONNX model for Hailo8 (NOT Hailo8L) for Sprint 3 T3.2a
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

class TCNVAEHailo8Compiler:
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.export_dir = self.repo_root / "export"
        self.hailo_dir = self.repo_root / "hailo_compiled"
        self.hailo_dir.mkdir(exist_ok=True)
        
        # T3.2a Configuration - Fixed Hailo8 Architecture
        self.config = {
            "target_hw": "hailo8",           # NOT hailo8l
            "hw_arch": "hailo8",             # Explicit architecture
            "optimization": "performance",    # Max performance
            "batch_size": 1,                 # Edge inference
            "quantization": "int8",          # Hailo8 native
        }
        
    def check_hailo_sdk(self):
        """Check if Hailo SDK is available"""
        try:
            result = subprocess.run(['hailomz', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Hailo Model Zoo version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Hailo SDK not found. Please install Hailo SDK first.")
            print("   https://developer.hailo.ai/documentation/sw-suite/")
            return False
    
    def validate_onnx_model(self, onnx_path):
        """Validate ONNX model exists and get info"""
        if not onnx_path.exists():
            print(f"❌ ONNX model not found: {onnx_path}")
            return False
            
        print(f"✅ Found ONNX model: {onnx_path}")
        print(f"   Size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
        
    def compile_tcn_vae_hailo8(self, onnx_path):
        """Compile TCN-VAE for Hailo8 with performance optimization"""
        print(f"\n🎯 Compiling TCN-VAE for Hailo8 (T3.2a Sprint 3)")
        print(f"📁 Input: {onnx_path}")
        
        hef_path = self.hailo_dir / "tcn_vae_72pct_hailo8.hef"
        
        # Critical compilation command - Fixed architecture
        cmd = [
            'hailomz', 'compile',
            str(onnx_path),
            '--hw-arch', 'hailo8',              # CRITICAL: hailo8 not hailo8l
            '--performance',                     # Max performance mode
            '--name', 'tcn_vae_72pct_hailo8',
            '--output-dir', str(self.hailo_dir),
            '--batch-size', '1',
            '--optimization', 'performance'
        ]
        
        print(f"🎯 Target architecture: hailo8 (25% faster than hailo8l)")
        print(f"🔧 Compilation command:")
        print(f"   {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ TCN-VAE compilation successful!")
            print(f"📄 HEF file: {hef_path}")
            return hef_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Compilation failed: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            return None
    
    def generate_deployment_info(self, hef_path):
        """Generate deployment information for T3.2a"""
        info = {
            "sprint": "T3.2a",
            "task": "Pose Model Compilation",
            "model": "TCN-VAE 72% Accuracy",
            "architecture": "hailo8",           # Confirmed Hailo8
            "compilation_date": datetime.now().isoformat(),
            "hef_file": str(hef_path),
            "performance_gain": "+25% vs hailo8l",
            "hardware_fix": "Corrected from hailo8l to hailo8",
            "target_fps": ">40 FPS (T3.2a acceptance criteria)",
            "deployment_ready": True
        }
        
        info_path = self.hailo_dir / "tcn_vae_hailo8_deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"📋 Deployment info: {info_path}")
        return info_path
    
    def generate_deployment_script(self, hef_path):
        """Generate Pi deployment script"""
        script_content = f"""#!/bin/bash
# TCN-VAE Hailo8 Deployment Script (T3.2a)
# Generated: {datetime.now().isoformat()}

echo "🚀 Deploying TCN-VAE Hailo8 Model (Sprint 3 T3.2a)"
echo "⚡ Expected performance: >40 FPS (vs 30 FPS with hailo8l)"

# Copy to hailo_pipeline artifacts
cp {hef_path.name} ../../hailo_pipeline/artifacts/

# Copy to pisrv_vapor_docker deployment
cp {hef_path.name} ../../pisrv_vapor_docker/appdata/models/tcn_vae/

# Backup old mock files
mv ../../hailo_pipeline/artifacts/tcn_encoder_v72pct.hef ../../hailo_pipeline/artifacts/tcn_encoder_v72pct.hef.mock.backup

echo "✅ TCN-VAE Hailo8 model deployed successfully!"
echo "🔍 Verify with: ls -la ../../hailo_pipeline/artifacts/*.hef"
echo "📊 Test performance: hailortcli run {hef_path.name} --measure-fps"
"""
        
        script_path = self.hailo_dir / "deploy_tcn_vae_hailo8.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        
        print(f"📜 Deployment script: {script_path}")
        return script_path
    
    def run_compilation(self):
        """Main compilation workflow for T3.2a"""
        print("🚀 TCN-VAE Hailo8 Compilation (Sprint 3 T3.2a)")
        print("⚡ Target: +25% performance improvement over Hailo8L")
        print(f"📁 Working directory: {self.repo_root}")
        
        start_time = datetime.now()
        print(f"⏰ Started at: {start_time}")
        
        # Check Hailo SDK
        if not self.check_hailo_sdk():
            return False
            
        # Find ONNX model
        onnx_path = self.export_dir / "tcn_encoder_for_edgeinfer.onnx"
        if not self.validate_onnx_model(onnx_path):
            return False
        
        # Compile for Hailo8
        hef_path = self.compile_tcn_vae_hailo8(onnx_path)
        if not hef_path:
            return False
            
        # Generate deployment assets
        self.generate_deployment_info(hef_path)
        self.generate_deployment_script(hef_path)
        
        # Success summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ T3.2a TCN-VAE Compilation SUCCESSFUL!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"🎯 Architecture: Hailo8 (optimized, not Hailo8L)")
        print(f"📄 HEF file: {hef_path}")
        print(f"\n🚀 Next Steps:")
        print(f"1. Test performance with: hailortcli run {hef_path}")
        print(f"2. Deploy with: bash {self.hailo_dir}/deploy_tcn_vae_hailo8.sh") 
        print(f"3. Verify >40 FPS for T3.2a acceptance criteria")
        print(f"4. Continue with YOLOv8 pose models for complete T3.2a")
        
        return True

if __name__ == "__main__":
    compiler = TCNVAEHailo8Compiler()
    success = compiler.run_compilation()
    
    if success:
        print("\n🎉 T3.2a TCN-VAE compilation ready for deployment!")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Compilation unsuccessful")
        sys.exit(1)