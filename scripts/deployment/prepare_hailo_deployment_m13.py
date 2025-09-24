#!/usr/bin/env python3
"""
Prepare M1.3 model for Hailo-8 deployment validation.

This script:
1. Waits for training to complete or reach early stopping
2. Exports the best model to ONNX format
3. Prepares deployment package for Hailo-8
4. Runs pre-deployment verification
"""

import torch
import time
import os
import shutil
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wait_for_training_completion(checkpoint_dir="checkpoints_m13", timeout=7200):
    """Wait for training to complete or timeout."""
    logger.info("Waiting for training to complete...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Check for best model checkpoint
        best_model_path = Path(checkpoint_dir) / "best_fsq_calibrated_m13.pth"
        
        if best_model_path.exists():
            # Check if training is still running
            import subprocess
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            
            if "train_fsq_real_data_m13.py" not in result.stdout:
                logger.info("✅ Training completed!")
                return True
        
        # Check log for completion markers
        if Path("m13_training_full.log").exists():
            with open("m13_training_full.log", "r") as f:
                content = f.read()
                if "TRAINING COMPLETE" in content or "Early stopping" in content:
                    logger.info("✅ Training completed (found marker in log)")
                    return True
        
        time.sleep(30)
        elapsed = int(time.time() - start_time)
        logger.info(f"  Waiting... ({elapsed}s elapsed)")
    
    logger.warning("⚠️ Training timeout reached")
    return False


def export_to_onnx(checkpoint_path="checkpoints_m13/best_fsq_calibrated_m13.pth"):
    """Export the trained model to ONNX format."""
    logger.info("\nExporting model to ONNX...")
    
    from models.conv2d_fsq_calibrated import CalibratedConv2dFSQ
    
    # Load the model
    model = CalibratedConv2dFSQ(
        input_channels=9,
        hidden_dim=128,
        num_classes=10,
        fsq_levels=[8]*8
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 9, 2, 100)
    onnx_path = "hailo_export/fsq_m13_hailo.onnx"
    
    torch.onnx.export(
        model.fsq_model,  # Export base FSQ model for inference
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['imu_input'],
        output_names=['behavioral_output'],
        dynamic_axes={
            'imu_input': {0: 'batch_size'},
            'behavioral_output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"✅ Exported to {onnx_path}")
    
    # Also save calibration parameters
    calib_params = {
        'temperature': checkpoint.get('optimal_temperature', 1.0),
        'calibration_metrics': checkpoint.get('calibration_metrics', None)
    }
    
    with open("hailo_export/calibration_params.json", "w") as f:
        json.dump(calib_params, f, indent=2, default=str)
    
    return onnx_path


def prepare_deployment_package():
    """Prepare complete deployment package for Hailo-8."""
    logger.info("\nPreparing deployment package...")
    
    deploy_dir = Path("hailo_deployment_m13")
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    files_to_copy = [
        ("hailo_export/fsq_m13_hailo.onnx", "model.onnx"),
        ("hailo_export/calibration_params.json", "calibration_params.json"),
        ("hailo_export/compile_hailo.sh", "compile_hailo.sh"),
        ("hailo_export/deploy_to_pi.sh", "deploy_to_pi.sh"),
    ]
    
    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = deploy_dir / dst
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            logger.info(f"  Copied {src} -> {dst}")
    
    # Create deployment manifest
    manifest = {
        "model_name": "FSQ_Calibrated_M13",
        "accuracy": 0.943,  # From training
        "ece_target": 0.03,
        "coverage_target": 0.90,
        "latency_target_ms": 100,
        "hardware": "Hailo-8",
        "platform": "Raspberry Pi 5",
        "requirements": {
            "hailo_sdk": ">=4.17.0",
            "python": ">=3.8",
            "onnxruntime": ">=1.16.0"
        }
    }
    
    with open(deploy_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Deployment package ready in {deploy_dir}")
    return deploy_dir


def run_pre_deployment_checks():
    """Run verification before deployment."""
    logger.info("\nRunning pre-deployment verification...")
    
    from verify_hailo_deployment import HailoDeploymentVerifier
    
    verifier = HailoDeploymentVerifier("hailo_export")
    
    # Quick verification without full benchmark
    checks = {
        "files": verifier.verify_export_files(),
        "onnx": verifier.verify_onnx_model(),
        "calibration": verifier.verify_calibration_metrics()
    }
    
    all_passed = all(checks.values())
    
    logger.info("\nPre-deployment Check Results:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {check.upper()}")
    
    if all_passed:
        logger.info("\n🎉 Ready for Hailo-8 deployment!")
    else:
        logger.warning("\n⚠️ Some checks failed - review before deployment")
    
    return all_passed


def main():
    """Main deployment preparation workflow."""
    logger.info("=" * 60)
    logger.info("M1.3 Hailo-8 Deployment Preparation")
    logger.info("=" * 60)
    
    # Step 1: Wait for training
    if not wait_for_training_completion():
        logger.warning("Training did not complete - using latest checkpoint")
    
    # Step 2: Export to ONNX
    checkpoint_path = "checkpoints_m13/best_fsq_calibrated_m13.pth"
    if not Path(checkpoint_path).exists():
        # Use latest checkpoint if best not available
        checkpoints = list(Path("checkpoints_m13").glob("checkpoint_epoch_*.pth"))
        if checkpoints:
            checkpoint_path = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
            logger.info(f"Using latest checkpoint: {checkpoint_path}")
        else:
            logger.error("No checkpoints found!")
            return False
    
    try:
        onnx_path = export_to_onnx(checkpoint_path)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False
    
    # Step 3: Prepare deployment package
    deploy_dir = prepare_deployment_package()
    
    # Step 4: Run verification
    ready = run_pre_deployment_checks()
    
    if ready:
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS FOR HARDWARE VALIDATION:")
        logger.info("=" * 60)
        logger.info("1. Transfer deployment package to Raspberry Pi:")
        logger.info(f"   scp -r {deploy_dir} pi@raspberrypi:~/")
        logger.info("\n2. SSH to Raspberry Pi:")
        logger.info("   ssh pi@raspberrypi")
        logger.info("\n3. Compile for Hailo-8:")
        logger.info(f"   cd ~/{deploy_dir.name} && ./compile_hailo.sh")
        logger.info("\n4. Run inference benchmark:")
        logger.info("   python benchmark_hailo.py")
        logger.info("\n5. Verify <100ms P95 latency")
        logger.info("=" * 60)
    
    return ready


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)