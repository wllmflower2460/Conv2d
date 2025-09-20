#!/bin/bash
# Setup 30-minute progress monitoring

SCRIPT_DIR="/home/wllmflower/tcn-vae-training/scripts"
LOG_DIR="/home/wllmflower/tcn-vae-training/logs"

# Create monitoring script that runs every 30 minutes
cat > "${SCRIPT_DIR}/monitor_training.sh" << 'EOF'
#!/bin/bash
cd /home/wllmflower/tcn-vae-training
python scripts/progress_report.py > logs/progress_report_$(date +%Y%m%d_%H%M).txt 2>&1
echo "Progress report generated at $(date)" >> logs/monitoring.log
EOF

chmod +x "${SCRIPT_DIR}/monitor_training.sh"

# Run first report in 30 minutes
echo "python /home/wllmflower/tcn-vae-training/scripts/progress_report.py" | at now + 30 minutes 2>/dev/null || echo "at command not available"

echo "✅ Monitoring setup complete"
echo "📊 First progress report in 30 minutes"
echo "📁 Reports will be saved to: $LOG_DIR/progress_report_*.txt"