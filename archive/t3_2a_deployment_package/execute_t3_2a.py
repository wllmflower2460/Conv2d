#!/usr/bin/env python3
"""
T3.2a Master Execution Script - Sprint 3 Pose Model Compilation
Orchestrates compilation of all models for T3.2a with Hailo8 architecture fix
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

class T3_2A_Master:
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.scripts_dir = self.repo_root / "scripts"
        self.hailo_dir = self.repo_root / "hailo_compiled"
        
        self.tasks = [
            {
                "name": "TCN-VAE 72% Model",
                "script": "compile_tcn_vae_hailo8.py",
                "priority": "P0",
                "status": "pending"
            },
            {
                "name": "YOLOv8 Pose Models", 
                "script": "compile_yolo_pose_hailo8_t3_2a.py",
                "priority": "P0",
                "status": "pending"
            }
        ]
    
    def print_banner(self):
        """Print T3.2a execution banner"""
        print("=" * 70)
        print("🚀 EXECUTING T3.2A: POSE MODEL COMPILATION")
        print("=" * 70)
        print("📋 Sprint 3 Task: T3.2a")
        print("🎯 Priority: P0 (Ready to Deploy)")
        print("⚡ Fix: Hailo8L → Hailo8 (+25% performance)")
        print("🏆 Goal: 3 HEF files ready, >40 FPS each")
        print("-" * 70)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Working: {self.repo_root}")
        print("=" * 70)
    
    def check_prerequisites(self):
        """Check if we have everything needed"""
        print("\\n🔍 Checking T3.2a Prerequisites...")
        
        # Check ONNX model exists
        onnx_path = self.repo_root / "export" / "tcn_encoder_for_edgeinfer.onnx"
        if onnx_path.exists():
            print(f"✅ TCN-VAE ONNX model: {onnx_path}")
        else:
            print(f"❌ TCN-VAE ONNX model missing: {onnx_path}")
            return False
            
        # Check compilation scripts exist
        for task in self.tasks:
            script_path = self.scripts_dir / task["script"]
            if script_path.exists():
                print(f"✅ Script ready: {task['script']}")
            else:
                print(f"❌ Script missing: {script_path}")
                return False
        
        # Check Hailo SDK (will show warning if not available)
        try:
            result = subprocess.run(['hailomz', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Hailo SDK available")
                return True
            else:
                print("⚠️  Hailo SDK not available locally")
                print("   Will create compilation package for remote execution")
                return True  # Continue anyway - we'll create deployment package
        except FileNotFoundError:
            print("⚠️  Hailo SDK not found locally")
            print("   Will create compilation package for remote execution")
            return True  # Continue anyway
    
    def execute_compilation_task(self, task):
        """Execute individual compilation task"""
        script_path = self.scripts_dir / task["script"]
        
        print(f"\\n🔄 Executing: {task['name']}")
        print(f"📜 Script: {task['script']}")
        
        try:
            # Run the compilation script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                print(f"✅ {task['name']}: SUCCESS")
                task["status"] = "completed"
                if result.stdout:
                    print("📤 Output:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ {task['name']}: FAILED")
                task["status"] = "failed" 
                if result.stderr:
                    print("📤 Error:")
                    print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {task['name']}: TIMEOUT")
            task["status"] = "timeout"
            return False
        except Exception as e:
            print(f"💥 {task['name']}: EXCEPTION - {e}")
            task["status"] = "error"
            return False
    
    def create_deployment_package(self):
        """Create deployment package for systems with Hailo SDK"""
        package_dir = self.repo_root / "t3_2a_deployment_package"
        package_dir.mkdir(exist_ok=True)
        
        print("\\n📦 Creating T3.2a Deployment Package...")
        
        # Copy compilation scripts
        scripts_package = package_dir / "scripts"
        scripts_package.mkdir(exist_ok=True)
        
        for task in self.tasks:
            src = self.scripts_dir / task["script"]
            dst = scripts_package / task["script"]
            if src.exists():
                dst.write_text(src.read_text())
                dst.chmod(0o755)
                print(f"📄 Packaged: {task['script']}")
        
        # Copy ONNX models
        models_package = package_dir / "models"
        models_package.mkdir(exist_ok=True)
        
        onnx_src = self.repo_root / "export" / "tcn_encoder_for_edgeinfer.onnx"
        if onnx_src.exists():
            onnx_dst = models_package / "tcn_encoder_for_edgeinfer.onnx"
            onnx_dst.write_bytes(onnx_src.read_bytes())
            print(f"📄 Packaged: TCN-VAE ONNX model")
        
        # Create deployment instructions
        instructions = f"""# T3.2a Deployment Package
## Sprint 3: Pose Model Compilation

### Quick Execution
```bash
# On system with Hailo SDK:
cd scripts/
python compile_tcn_vae_hailo8.py
python compile_yolo_pose_hailo8_t3_2a.py
```

### Requirements
- Hailo SDK installed (hailomz command available)
- Internet connection (for YOLO model downloads)
- Python 3.7+ with requests module

### Expected Output
- tcn_vae_72pct_hailo8.hef
- yolov8n_pose_human_hailo8.hef  
- yolov8s_pose_dog_hailo8.hef

### T3.2a Acceptance Criteria
✅ 3 HEF files compiled with --hw-arch hailo8
✅ Each model achieving >40 FPS
✅ 25% performance improvement vs hailo8l

Generated: {datetime.now().isoformat()}
"""
        
        readme_path = package_dir / "README.md"
        readme_path.write_text(instructions)
        
        # Create archive if tar is available
        try:
            import tarfile
            archive_path = self.repo_root / "t3_2a_deployment_package.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(package_dir, arcname='t3_2a_deployment_package')
            print(f"📦 Archive created: {archive_path}")
        except Exception as e:
            print(f"⚠️  Archive creation failed: {e}")
            
        print(f"✅ Deployment package ready: {package_dir}")
        return package_dir
    
    def generate_t3_2a_report(self):
        """Generate final T3.2a execution report"""
        completed_tasks = [t for t in self.tasks if t["status"] == "completed"]
        failed_tasks = [t for t in self.tasks if t["status"] != "completed"]
        
        report = {
            "sprint_task": "T3.2a: Pose Model Compilation",
            "execution_date": datetime.now().isoformat(),
            "priority": "P0 (Ready to Deploy)",
            "architecture_fix": "hailo8l → hailo8 (+25% performance)",
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": f"{len(completed_tasks)/len(self.tasks)*100:.1f}%",
            "tasks": self.tasks,
            "next_actions": []
        }
        
        if len(completed_tasks) == len(self.tasks):
            report["status"] = "T3.2a COMPLETE"
            report["next_actions"] = [
                "Deploy HEF files to hailo_pipeline/artifacts/",
                "Test performance: hailortcli run model.hef --measure-fps",
                "Verify >40 FPS acceptance criteria",
                "Proceed to T3.3a: Multi-Model Scheduler"
            ]
        else:
            report["status"] = "T3.2a PARTIAL"
            report["next_actions"] = [
                "Execute deployment package on system with Hailo SDK",
                "Complete remaining model compilations",
                "Validate all HEF files before T3.3a"
            ]
        
        report_path = self.repo_root / "T3_2A_EXECUTION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report, report_path
    
    def run_t3_2a(self):
        """Main T3.2a execution workflow"""
        self.print_banner()
        
        # Prerequisites check
        if not self.check_prerequisites():
            print("❌ Prerequisites failed - cannot continue")
            return False
        
        # Execute compilation tasks
        print("\\n🚀 Starting T3.2a Model Compilation...")
        
        success_count = 0
        for task in self.tasks:
            if self.execute_compilation_task(task):
                success_count += 1
        
        # Create deployment package
        package_dir = self.create_deployment_package()
        
        # Generate report
        report, report_path = self.generate_t3_2a_report()
        
        # Final summary
        print("\\n" + "=" * 70)
        print("📊 T3.2A EXECUTION SUMMARY")
        print("=" * 70)
        print(f"✅ Tasks completed: {success_count}/{len(self.tasks)}")
        print(f"📋 Status: {report['status']}")
        print(f"📄 Report: {report_path}")
        print(f"📦 Package: {package_dir}")
        
        if success_count == len(self.tasks):
            print("\\n🎉 T3.2A COMPLETE! All models compiled with Hailo8 architecture.")
            print("🚀 Ready to proceed to T3.3a: Multi-Model Scheduler")
        else:
            print(f"\\n⚠️  T3.2A PARTIAL: Use deployment package for remaining compilations")
            
        print("=" * 70)
        return success_count > 0

if __name__ == "__main__":
    master = T3_2A_Master()
    success = master.run_t3_2a()
    sys.exit(0 if success else 1)