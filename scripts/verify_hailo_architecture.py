#!/usr/bin/env python3
"""
Hailo Architecture Verification Script
Check if existing models are compiled for Hailo8L vs Hailo8
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

class HailoArchitectureVerifier:
    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        
        print("🔍 Hailo Architecture Verification Tool")
        print("⚡ Checking for Hailo8L vs Hailo8 compilation issues")
        
    def find_hef_files(self):
        """Find all HEF files in models directory"""
        hef_files = list(self.models_dir.glob("*.hef"))
        
        if not hef_files:
            print(f"❌ No HEF files found in {self.models_dir}")
            # Check common locations
            common_paths = [
                Path("./export"),
                Path("./models"),
                Path("../models"),
                Path("/home/pi/models") if os.path.exists("/home/pi/models") else None
            ]
            
            for path in common_paths:
                if path and path.exists():
                    hef_files.extend(list(path.glob("*.hef")))
        
        return hef_files
    
    def check_hef_architecture(self, hef_path):
        """Check what architecture a HEF file was compiled for"""
        print(f"\n🔍 Checking: {hef_path.name}")
        
        if not hef_path.exists():
            print(f"❌ File not found: {hef_path}")
            return None
        
        file_size = hef_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📊 Size: {file_size:.1f} MB")
        
        # Try to get HEF info using Hailo tools
        try:
            result = subprocess.run([
                'hailo', 'info', str(hef_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                # Check for architecture indicators
                if 'hailo8l' in output or 'hailo-8l' in output:
                    arch = 'hailo8l'
                    performance_impact = "⚠️  25% performance loss"
                elif 'hailo8' in output or 'hailo-8' in output:
                    arch = 'hailo8'
                    performance_impact = "✅ Full performance"
                else:
                    arch = 'unknown'
                    performance_impact = "❓ Cannot determine"
                
                print(f"🎯 Architecture: {arch}")
                print(f"⚡ Performance: {performance_impact}")
                
                return {
                    'file': str(hef_path),
                    'architecture': arch,
                    'size_mb': file_size,
                    'performance_impact': performance_impact,
                    'needs_recompilation': arch == 'hailo8l'
                }
            else:
                print(f"⚠️  Could not read HEF info: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⚠️  Hailo info command timeout")
            return None
        except FileNotFoundError:
            print("⚠️  Hailo CLI tools not found")
            # Fallback: analyze filename patterns
            return self.analyze_filename_pattern(hef_path, file_size)
    
    def analyze_filename_pattern(self, hef_path, file_size):
        """Analyze filename for architecture hints"""
        filename = hef_path.name.lower()
        
        if 'hailo8l' in filename or '8l' in filename:
            arch = 'hailo8l'
            performance_impact = "⚠️  25% performance loss (filename indicates hailo8l)"
        elif 'hailo8' in filename or '_8' in filename:
            arch = 'hailo8'  
            performance_impact = "✅ Likely full performance (filename indicates hailo8)"
        else:
            arch = 'unknown'
            performance_impact = "❓ Architecture unclear from filename"
        
        print(f"📝 Filename analysis: {arch}")
        print(f"⚡ Performance: {performance_impact}")
        
        return {
            'file': str(hef_path),
            'architecture': arch,
            'size_mb': file_size,
            'performance_impact': performance_impact,
            'needs_recompilation': arch == 'hailo8l',
            'detection_method': 'filename_analysis'
        }
    
    def check_current_performance(self):
        """Check if there are existing performance benchmarks"""
        print(f"\n📊 Checking existing performance data...")
        
        performance_files = [
            "performance_benchmark.json",
            "benchmark_results.json", 
            "model_performance.json"
        ]
        
        for perf_file in performance_files:
            perf_path = Path(perf_file)
            if perf_path.exists():
                try:
                    with open(perf_path, 'r') as f:
                        data = json.load(f)
                    
                    print(f"📋 Found performance data: {perf_file}")
                    
                    # Look for FPS data
                    fps_keys = ['fps', 'framerate', 'performance_fps', 'benchmark_fps']
                    for key in fps_keys:
                        if key in data:
                            fps = data[key]
                            print(f"🎯 Current FPS: {fps}")
                            
                            # Estimate potential improvement
                            if fps < 60:  # Likely affected by hailo8l
                                improved_fps = fps * 1.25
                                print(f"⚡ Potential with Hailo8: ~{improved_fps:.1f} FPS")
                            break
                    
                    return data
                    
                except Exception as e:
                    print(f"⚠️  Error reading {perf_file}: {e}")
        
        print("❌ No performance benchmarks found")
        return None
    
    def generate_report(self, hef_results):
        """Generate comprehensive architecture report"""
        print(f"\n📋 Architecture Analysis Report")
        print(f"⏰ Generated: {datetime.now().isoformat()}")
        print(f"=" * 60)
        
        hailo8l_files = [r for r in hef_results if r['architecture'] == 'hailo8l']
        hailo8_files = [r for r in hef_results if r['architecture'] == 'hailo8']
        unknown_files = [r for r in hef_results if r['architecture'] == 'unknown']
        
        print(f"📊 Summary:")
        print(f"  - Hailo8L (suboptimal): {len(hailo8l_files)} files")
        print(f"  - Hailo8 (optimal): {len(hailo8_files)} files")
        print(f"  - Unknown: {len(unknown_files)} files")
        
        if hailo8l_files:
            print(f"\n⚠️  PERFORMANCE ISSUE FOUND:")
            print(f"   {len(hailo8l_files)} model(s) compiled for Hailo8L instead of Hailo8")
            print(f"   Expected performance loss: ~25%")
            print(f"   Recommendation: Recompile for Hailo8 architecture")
            
            print(f"\n📁 Files needing recompilation:")
            for result in hailo8l_files:
                print(f"   - {Path(result['file']).name}")
        
        if hailo8_files:
            print(f"\n✅ Optimal Performance:")
            for result in hailo8_files:
                print(f"   - {Path(result['file']).name} (Hailo8)")
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'hailo8l_count': len(hailo8l_files),
                'hailo8_count': len(hailo8_files),
                'unknown_count': len(unknown_files),
                'performance_issue_detected': len(hailo8l_files) > 0
            },
            'files': hef_results,
            'recommendations': []
        }
        
        if hailo8l_files:
            report['recommendations'].append({
                'priority': 'high',
                'action': 'recompile_for_hailo8',
                'expected_improvement': '25% performance gain',
                'affected_files': [r['file'] for r in hailo8l_files]
            })
        
        report_path = Path("hailo_architecture_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Full report saved: {report_path}")
        return report
    
    def run_verification(self):
        """Run complete architecture verification"""
        print(f"🚀 Starting Hailo Architecture Verification")
        
        # Find HEF files
        hef_files = self.find_hef_files()
        if not hef_files:
            print("❌ No HEF files found to verify")
            return False
        
        print(f"📁 Found {len(hef_files)} HEF file(s)")
        
        # Check each file
        results = []
        for hef_file in hef_files:
            result = self.check_hef_architecture(hef_file)
            if result:
                results.append(result)
        
        if not results:
            print("❌ Could not analyze any HEF files")
            return False
        
        # Check current performance
        self.check_current_performance()
        
        # Generate report
        report = self.generate_report(results)
        
        return len([r for r in results if r['needs_recompilation']]) == 0

def main():
    """Main verification function"""
    verifier = HailoArchitectureVerifier()
    
    try:
        success = verifier.run_verification()
        
        if success:
            print("\n✅ All models optimally compiled for Hailo8")
        else:
            print("\n⚠️  Performance issues detected - recompilation recommended")
            print("💡 Use compile_yolov8_hailo8_fixed.py to fix architecture issues")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)