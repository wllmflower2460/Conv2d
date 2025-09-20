#!/usr/bin/env python3
"""
Run progress report in 30 minutes
"""
import time
import subprocess
from datetime import datetime

print(f"⏰ Waiting 30 minutes for progress report...")
print(f"Started at: {datetime.now()}")

# Wait 30 minutes (1800 seconds)
time.sleep(1800)

print(f"🔍 Running progress report at: {datetime.now()}")

# Run the progress report
subprocess.run([
    'python', 
    '/home/wllmflower/tcn-vae-training/scripts/progress_report.py'
])

print("📊 Progress report completed!")