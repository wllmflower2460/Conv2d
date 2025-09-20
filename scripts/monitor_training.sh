#!/bin/bash
cd /home/wllmflower/tcn-vae-training
python scripts/progress_report.py > logs/progress_report_$(date +%Y%m%d_%H%M).txt 2>&1
echo "Progress report generated at $(date)" >> logs/monitoring.log
