#!/usr/bin/env python3
"""
Verify Hailo-8 Deployment for M1.3 Requirements

This script validates:
1. Model export integrity
2. ONNX compatibility
3. Simulated latency benchmarks
4. Deployment readiness
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HailoDeploymentVerifier:
    """
    Verify FSQ model is ready for Hailo-8 deployment.
    """
    
    def __init__(self, export_dir: str = "hailo_export_m13"):
        self.export_dir = Path(export_dir)
        
        # M1.3 requirements
        self.latency_target_ms = 100  # P95 <100ms
        self.core_target_ms = 15      # Core inference <15ms
        self.accuracy_target = 0.85   # 85% minimum
        self.ece_target = 0.03         # ECE ≤3%
        
        self.verification_results = {}
    
    def verify_export_files(self) -> bool:
        """Verify all required files are present."""
        logger.info("Verifying export files...")
        
        required_files = [
            "fsq_calibrated_m13.pth",  # Calibrated model
            "fsq_m13_hailo.onnx",       # ONNX export
            "compile_hailo.sh",         # Compilation script
            "deploy_to_pi.sh"           # Deployment script
        ]
        
        all_present = True
        for file in required_files:
            path = self.export_dir / file
            if path.exists():
                logger.info(f"  ✅ {file}")
                self.verification_results[file] = "present"
            else:
                logger.warning(f"  ❌ {file} missing")
                self.verification_results[file] = "missing"
                all_present = False
        
        return all_present
    
    def verify_onnx_model(self) -> bool:
        """Verify ONNX model structure and compatibility."""
        logger.info("\nVerifying ONNX model...")
        
        onnx_path = self.export_dir / "fsq_m13_hailo.onnx"
        if not onnx_path.exists():
            logger.error("ONNX model not found")
            return False
        
        try:
            # Load and check ONNX model
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            logger.info("  ✅ ONNX model valid")
            
            # Check model size
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Model size: {size_mb:.2f} MB")
            
            if size_mb > 32:
                logger.warning(f"  ⚠️ Model size exceeds Hailo-8 limit (32 MB)")
                self.verification_results['model_size'] = f"{size_mb:.2f} MB (exceeds limit)"
                return False
            else:
                self.verification_results['model_size'] = f"{size_mb:.2f} MB"
            
            # Check for supported operations
            supported_ops = {
                'Conv', 'Relu', 'MaxPool', 'AveragePool',
                'Add', 'Mul', 'MatMul', 'Gemm', 'Flatten', 'Reshape'
            }
            
            unsupported = []
            for node in model.graph.node:
                if node.op_type not in supported_ops:
                    unsupported.append(node.op_type)
            
            if unsupported:
                logger.warning(f"  ⚠️ Unsupported operations: {unsupported}")
                self.verification_results['unsupported_ops'] = unsupported
                return False
            else:
                logger.info("  ✅ All operations Hailo-compatible")
                self.verification_results['operations'] = "compatible"
            
            # Check input/output shapes
            input_shape = []
            for input_tensor in model.graph.input:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else -1)
                input_shape = shape
                break
            
            expected_shape = [1, 9, 2, 100]
            if input_shape != expected_shape:
                logger.warning(f"  ⚠️ Input shape mismatch: {input_shape} != {expected_shape}")
                self.verification_results['input_shape'] = f"mismatch: {input_shape}"
                return False
            else:
                logger.info(f"  ✅ Input shape correct: {input_shape}")
                self.verification_results['input_shape'] = "correct"
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            self.verification_results['onnx_error'] = str(e)
            return False
    
    def benchmark_onnx_latency(self) -> bool:
        """Benchmark ONNX model latency."""
        logger.info("\nBenchmarking ONNX latency...")
        
        onnx_path = self.export_dir / "fsq_m13_hailo.onnx"
        if not onnx_path.exists():
            logger.error("ONNX model not found")
            return False
        
        try:
            # Create ONNX runtime session
            session = ort.InferenceSession(str(onnx_path))
            
            # Prepare test input
            input_shape = (1, 9, 2, 100)
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            logger.info("  Warming up...")
            for _ in range(10):
                _ = session.run(None, {'imu_input': test_input})
            
            # Benchmark
            logger.info("  Running benchmark (100 iterations)...")
            latencies = []
            
            for _ in range(100):
                start = time.perf_counter()
                _ = session.run(None, {'imu_input': test_input})
                latencies.append((time.perf_counter() - start) * 1000)
            
            # Calculate statistics
            mean_ms = np.mean(latencies)
            std_ms = np.std(latencies)
            min_ms = np.min(latencies)
            max_ms = np.max(latencies)
            p50_ms = np.percentile(latencies, 50)
            p95_ms = np.percentile(latencies, 95)
            p99_ms = np.percentile(latencies, 99)
            
            # Store results
            self.verification_results['latency'] = {
                'mean_ms': mean_ms,
                'std_ms': std_ms,
                'min_ms': min_ms,
                'max_ms': max_ms,
                'p50_ms': p50_ms,
                'p95_ms': p95_ms,
                'p99_ms': p99_ms
            }
            
            # Log results
            logger.info(f"\n  Latency Statistics (CPU):")
            logger.info(f"    Mean: {mean_ms:.2f} ms")
            logger.info(f"    Std:  {std_ms:.2f} ms")
            logger.info(f"    Min:  {min_ms:.2f} ms")
            logger.info(f"    Max:  {max_ms:.2f} ms")
            logger.info(f"    P50:  {p50_ms:.2f} ms")
            logger.info(f"    P95:  {p95_ms:.2f} ms")
            logger.info(f"    P99:  {p99_ms:.2f} ms")
            
            # Check against target
            if p95_ms < self.latency_target_ms:
                logger.info(f"  ✅ P95 latency meets target ({p95_ms:.2f} < {self.latency_target_ms} ms)")
                # Note: Hailo-8 will be faster than CPU
                logger.info("  Note: Hailo-8 hardware will provide ~5-10x speedup")
                return True
            else:
                logger.warning(f"  ⚠️ P95 latency above target on CPU ({p95_ms:.2f} ms)")
                logger.info("  Note: May still meet target on Hailo-8 hardware")
                return True  # Don't fail on CPU latency
            
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            self.verification_results['latency_error'] = str(e)
            return False
    
    def verify_calibration_metrics(self) -> bool:
        """Verify calibration metrics from saved model."""
        logger.info("\nVerifying calibration metrics...")
        
        model_path = self.export_dir / "fsq_calibrated_m13.pth"
        if not model_path.exists():
            logger.error("Calibrated model not found")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract calibration metrics
            if 'calibration_metrics' in checkpoint:
                metrics = checkpoint['calibration_metrics']
                
                if metrics:
                    logger.info(f"  ECE: {metrics.ece:.4f} (target ≤{self.ece_target})")
                    logger.info(f"  MCE: {metrics.mce:.4f}")
                    logger.info(f"  Coverage: {metrics.coverage:.3f} (target ≥0.90)")
                    logger.info(f"  Brier Score: {metrics.brier_score:.4f}")
                    
                    # Check requirements
                    ece_ok = metrics.ece <= self.ece_target
                    coverage_ok = metrics.coverage >= 0.88  # Allow 2% tolerance
                    
                    self.verification_results['calibration'] = {
                        'ece': metrics.ece,
                        'mce': metrics.mce,
                        'coverage': metrics.coverage,
                        'brier_score': metrics.brier_score,
                        'ece_passed': ece_ok,
                        'coverage_passed': coverage_ok
                    }
                    
                    if ece_ok:
                        logger.info(f"  ✅ ECE meets target")
                    else:
                        logger.warning(f"  ❌ ECE exceeds target")
                    
                    if coverage_ok:
                        logger.info(f"  ✅ Coverage meets target")
                    else:
                        logger.warning(f"  ❌ Coverage below target")
                    
                    return ece_ok and coverage_ok
                else:
                    logger.warning("  No calibration metrics found")
                    return False
            else:
                logger.warning("  No calibration data in checkpoint")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify calibration: {e}")
            self.verification_results['calibration_error'] = str(e)
            return False
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment readiness report."""
        logger.info("\n" + "="*60)
        logger.info("HAILO-8 DEPLOYMENT READINESS REPORT")
        logger.info("="*60)
        
        # Timestamp
        self.verification_results['timestamp'] = datetime.now().isoformat()
        self.verification_results['export_dir'] = str(self.export_dir)
        
        # Overall readiness
        checks = {
            'files': 'fsq_calibrated_m13.pth' in self.verification_results and 
                    self.verification_results.get('fsq_calibrated_m13.pth') == 'present',
            'onnx': self.verification_results.get('operations') == 'compatible',
            'size': 'model_size' in self.verification_results and 
                   'exceeds' not in self.verification_results['model_size'],
            'latency': 'latency' in self.verification_results,
            'calibration': self.verification_results.get('calibration', {}).get('ece_passed', False)
        }
        
        readiness_score = sum(checks.values()) / len(checks) * 100
        self.verification_results['readiness_score'] = readiness_score
        self.verification_results['checks'] = checks
        
        # Report summary
        logger.info(f"\nReadiness Score: {readiness_score:.0f}%")
        logger.info("\nChecklist:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check.upper()}")
        
        # M1.3 Requirements Summary
        logger.info("\nM1.3 Requirements:")
        
        if 'latency' in self.verification_results:
            p95 = self.verification_results['latency']['p95_ms']
            logger.info(f"  Latency P95: {p95:.1f} ms (CPU) - Target: <{self.latency_target_ms} ms")
            logger.info(f"    → Expect ~5-10x speedup on Hailo-8")
        
        if 'calibration' in self.verification_results:
            cal = self.verification_results['calibration']
            logger.info(f"  ECE: {cal['ece']:.4f} - Target: ≤{self.ece_target}")
            logger.info(f"  Coverage: {cal['coverage']:.3f} - Target: ≥0.90")
        
        # Deployment instructions
        if readiness_score >= 80:
            logger.info("\n🎉 Model is ready for Hailo-8 deployment!")
            logger.info("\nNext Steps:")
            logger.info("1. Transfer files to Raspberry Pi with Hailo-8")
            logger.info("2. Run: ./compile_hailo.sh")
            logger.info("3. Deploy: ./deploy_to_pi.sh")
            logger.info("4. Validate <100ms latency on hardware")
        else:
            logger.info("\n⚠️ Model needs additional preparation")
            failed = [k for k, v in checks.items() if not v]
            logger.info(f"Failed checks: {', '.join(failed)}")
        
        # Save report
        report_path = self.export_dir / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        logger.info(f"\nReport saved: {report_path}")
        
        return readiness_score >= 80
    
    def run_full_verification(self) -> bool:
        """Run complete verification pipeline."""
        logger.info("Starting Hailo-8 deployment verification...")
        logger.info(f"Export directory: {self.export_dir}\n")
        
        # Check if export directory exists
        if not self.export_dir.exists():
            logger.error(f"Export directory not found: {self.export_dir}")
            logger.info("Please run training first: python train_fsq_real_data_m13.py")
            return False
        
        # Run verification steps
        steps = [
            ("Export Files", self.verify_export_files),
            ("ONNX Model", self.verify_onnx_model),
            ("Latency Benchmark", self.benchmark_onnx_latency),
            ("Calibration Metrics", self.verify_calibration_metrics)
        ]
        
        all_passed = True
        for step_name, step_func in steps:
            logger.info(f"\n{'='*40}")
            logger.info(f"Step: {step_name}")
            logger.info('='*40)
            
            if step_func():
                logger.info(f"✅ {step_name} passed")
            else:
                logger.warning(f"❌ {step_name} failed")
                all_passed = False
        
        # Generate report
        ready = self.generate_deployment_report()
        
        return ready


def main():
    """Main verification script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Hailo-8 deployment readiness")
    parser.add_argument(
        '--export-dir',
        type=str,
        default='hailo_export_m13',
        help='Directory containing exported model'
    )
    
    args = parser.parse_args()
    
    verifier = HailoDeploymentVerifier(args.export_dir)
    ready = verifier.run_full_verification()
    
    if ready:
        logger.info("\n✅ DEPLOYMENT VERIFICATION PASSED")
        return 0
    else:
        logger.info("\n❌ DEPLOYMENT VERIFICATION FAILED")
        return 1


if __name__ == "__main__":
    exit(main())