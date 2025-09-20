#!/usr/bin/env python3
"""
30-Minute Progress Report Generator
"""

import json
import os
import sys
from datetime import datetime, timedelta
import subprocess
import glob

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.training_config import TrainingConfig

def get_gpu_status():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'utilization': f"{gpu_util}%",
                'memory': f"{mem_used}MB / {mem_total}MB",
                'temperature': f"{temp}°C"
            }
    except Exception as e:
        return {'error': str(e)}
    return {'status': 'GPU info unavailable'}

def parse_training_logs():
    """Parse the latest training logs"""
    log_file = '/home/wllmflower/tcn-vae-training/logs/overnight_training.jsonl'
    
    if not os.path.exists(log_file):
        return {'error': 'No training logs found'}
    
    logs = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
    except Exception as e:
        return {'error': f'Error reading logs: {e}'}
    
    if not logs:
        return {'error': 'No log entries found'}
    
    # Get latest entries (last 30 minutes worth)
    now = datetime.now()
    recent_logs = []
    
    for log in logs:
        try:
            log_time = datetime.strptime(log['timestamp'], "%Y-%m-%d %H:%M:%S")
            if (now - log_time).total_seconds() <= 1800:  # 30 minutes
                recent_logs.append(log)
        except:
            continue
    
    return {
        'total_epochs': len(logs),
        'recent_epochs': len(recent_logs),
        'latest_entry': logs[-1] if logs else None,
        'best_accuracy': max([log['best_so_far'] for log in logs]) if logs else 0,
        'current_accuracy': logs[-1]['val_accuracy'] if logs else 0,
        'recent_trend': recent_logs[-5:] if len(recent_logs) >= 5 else recent_logs
    }

def get_model_files():
    """Check for saved model files"""
    model_dir = '/home/wllmflower/tcn-vae-training/models'
    files = []
    
    if os.path.exists(model_dir):
        for pattern in ['*overnight*.pth', '*best*.pth', 'checkpoint*.pth']:
            files.extend(glob.glob(os.path.join(model_dir, pattern)))
    
    file_info = []
    for f in files:
        stat = os.stat(f)
        file_info.append({
            'name': os.path.basename(f),
            'size_mb': stat.st_size / (1024*1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return sorted(file_info, key=lambda x: x['modified'], reverse=True)

def generate_report():
    """Generate comprehensive progress report"""
    print("🔍 TCN-VAE Training Progress Report")
    print("=" * 50)
    print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # GPU Status
    print("🖥️  GPU STATUS")
    print("-" * 20)
    gpu_info = get_gpu_status()
    for key, value in gpu_info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
    
    # Training Progress
    print("📊 TRAINING PROGRESS")
    print("-" * 20)
    log_data = parse_training_logs()
    
    if 'error' in log_data:
        print(f"  ❌ {log_data['error']}")
    else:
        print(f"  Total Epochs Completed: {log_data['total_epochs']}")
        print(f"  Epochs in Last 30min: {log_data['recent_epochs']}")
        print(f"  Current Best Accuracy: {log_data['best_accuracy']:.4f}")
        print(f"  Current Accuracy: {log_data['current_accuracy']:.4f}")
        
        if log_data['latest_entry']:
            latest = log_data['latest_entry']
            print(f"  Latest Epoch: {latest['epoch']}")
            print(f"  Latest Loss: {latest['val_loss']:.4f}")
            print(f"  Learning Rate: {latest['learning_rate']:.6f}")
            print(f"  Epoch Time: {latest['epoch_time']:.1f}s")
            
            # Progress vs target
            target = 0.6  # 60% target
            current_best = log_data['best_accuracy']
            progress_pct = (current_best / target) * 100
            print(f"  Progress to 60% Target: {progress_pct:.1f}%")
            
            # Improvement over baseline - use configuration
            baseline = TrainingConfig.BASELINE_ACCURACY
            if current_best > baseline:
                improvement = ((current_best / baseline) - 1) * 100
                print(f"  Improvement vs Baseline: +{improvement:.1f}%")
    
    print()
    
    # Recent Trend Analysis
    if 'recent_trend' in log_data and log_data['recent_trend']:
        print("📈 RECENT TREND (Last 5 epochs)")
        print("-" * 35)
        for i, entry in enumerate(log_data['recent_trend']):
            status = "🔥" if entry['is_best'] else "  "
            print(f"  {status} Epoch {entry['epoch']}: {entry['val_accuracy']:.4f} (Loss: {entry['val_loss']:.4f})")
    
    print()
    
    # Saved Models
    print("💾 SAVED MODELS")
    print("-" * 15)
    models = get_model_files()
    if models:
        for model in models[:5]:  # Show top 5 recent
            print(f"  📁 {model['name']} ({model['size_mb']:.1f}MB) - {model['modified']}")
    else:
        print("  No model files found yet")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 20)
    if 'error' not in log_data and log_data['latest_entry']:
        latest = log_data['latest_entry']
        current_acc = latest['val_accuracy']
        
        if current_acc > 0.6:
            print("  🎉 Target exceeded! Consider early stopping or exploration")
        elif current_acc > 0.58:
            print("  🎯 Close to target! Continue training")
        elif current_acc < 0.5:
            print("  ⚠️  Below baseline - check for issues")
        else:
            print("  📈 Making progress - continue monitoring")
        
        # Learning rate check
        if latest['learning_rate'] < 1e-5:
            print("  📉 Learning rate very low - may need restart")
        elif latest['learning_rate'] > 1e-3:
            print("  📈 Learning rate high - monitor for instability")
    else:
        print("  🔍 Waiting for training data...")
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    generate_report()