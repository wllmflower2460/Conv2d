#!/usr/bin/env python3
"""Monitor M1.3 training progress."""

import time
import re
import os
from datetime import datetime

def parse_log_line(line):
    """Parse training metrics from log line."""
    patterns = {
        'epoch': r'Epoch (\d+)/(\d+)',
        'train_loss': r'Train: Loss=([\d.]+)',
        'train_acc': r'Acc=([\d.]+)',
        'val_loss': r'Val:\s+Loss=([\d.]+)',
        'val_acc': r'Val:.*Acc=([\d.]+)',
        'codes': r'Codes: Used=(\d+)',
        'perplexity': r'Perp=([\d.]+)',
        'best_acc': r'Best validation accuracy: ([\d.]+)',
        'ece': r'ECE:\s+([\d.]+)',
        'coverage': r'Coverage:\s+([\d.]+)',
    }
    
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            if key == 'epoch':
                metrics['epoch'] = int(match.group(1))
                metrics['total_epochs'] = int(match.group(2))
            else:
                metrics[key] = float(match.group(1))
    
    return metrics

def monitor_training(log_file='m13_training_full.log', update_interval=10):
    """Monitor training progress from log file."""
    print("=" * 60)
    print("M1.3 FSQ Calibrated Model Training Monitor")
    print("=" * 60)
    print(f"Log file: {log_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    last_position = 0
    current_metrics = {}
    
    while True:
        try:
            if not os.path.exists(log_file):
                print(f"Waiting for {log_file}...")
                time.sleep(update_interval)
                continue
            
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
                
                for line in new_lines:
                    # Check for training completion
                    if "TRAINING COMPLETE" in line:
                        print("\n🎉 Training completed!")
                        return
                    
                    if "DEPLOYMENT READY" in line:
                        print("\n✅ Model is ready for deployment!")
                        
                    # Parse metrics
                    metrics = parse_log_line(line)
                    current_metrics.update(metrics)
                    
                    # Display progress
                    if 'epoch' in metrics:
                        print(f"\n📊 Epoch {metrics['epoch']}/{metrics.get('total_epochs', '?')}")
                    
                    if 'train_acc' in metrics:
                        print(f"  Train Acc: {metrics['train_acc']:.1%}")
                        
                    if 'val_acc' in metrics:
                        print(f"  Val Acc:   {metrics['val_acc']:.1%}")
                        
                        # Check if meeting M1.3 target
                        if metrics['val_acc'] >= 0.85:
                            print(f"  🎯 Reached M1.3 accuracy target!")
                    
                    if 'codes' in metrics:
                        print(f"  FSQ Codes: {int(metrics['codes'])}/512 used")
                        
                    if 'best_acc' in current_metrics:
                        print(f"  Best:      {current_metrics['best_acc']:.1%}")
                    
                    if 'ece' in metrics:
                        print(f"\n📈 Calibration Metrics:")
                        print(f"  ECE:      {metrics['ece']:.4f} {'✅' if metrics['ece'] <= 0.03 else '❌'}")
                        
                    if 'coverage' in metrics:
                        print(f"  Coverage: {metrics['coverage']:.1%} {'✅' if metrics['coverage'] >= 0.88 else '❌'}")
                    
                    # Check for warnings or errors
                    if "WARNING" in line or "⚠️" in line:
                        print(f"\n⚠️  {line.strip()}")
                    
                    if "ERROR" in line or "❌" in line:
                        print(f"\n❌ {line.strip()}")
                        
                    if "✅ PASS" in line:
                        print(f"  {line.strip()}")
            
            # Show current status
            if current_metrics:
                print(f"\r⏱️  Last update: {datetime.now().strftime('%H:%M:%S')} | "
                      f"Epoch: {current_metrics.get('epoch', '?')}/{current_metrics.get('total_epochs', '?')} | "
                      f"Best: {current_metrics.get('best_acc', 0):.1%}", end='', flush=True)
            
            time.sleep(update_interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(update_interval)

if __name__ == "__main__":
    monitor_training()