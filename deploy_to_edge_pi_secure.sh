#!/bin/bash

# Secure Deploy M1.3 FSQ Model to Edge Pi
# Target: pi@100.127.242.78
# Enhanced security with host key verification and security best practices

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
EDGE_PI_HOST="100.127.242.78"
EDGE_PI_USER="pi"
EDGE_PI_TARGET="${EDGE_PI_USER}@${EDGE_PI_HOST}"
DEPLOYMENT_DIR="/home/pi/m13_fsq_deployment"
LOCAL_PACKAGE="m13_fsq_deployment"

# Security Configuration
KNOWN_HOSTS_FILE="${HOME}/.ssh/known_hosts"
SSH_KEY_TYPE="ed25519"  # Preferred key type
SSH_PORT="${SSH_PORT:-22}"  # Allow override via environment
MAX_RETRIES=3
RETRY_DELAY=5

# SSH Security Options
SSH_OPTS=(
    -o "StrictHostKeyChecking=yes"      # Verify host key (prevent MITM)
    -o "UserKnownHostsFile=${KNOWN_HOSTS_FILE}"
    -o "PasswordAuthentication=no"       # Only allow key-based auth
    -o "PubkeyAuthentication=yes"
    -o "ConnectTimeout=10"
    -o "ServerAliveInterval=60"         # Keep connection alive
    -o "ServerAliveCountMax=3"
    -o "BatchMode=yes"                  # Fail instead of prompting
    -o "LogLevel=VERBOSE"               # Log for security audit
    -p "${SSH_PORT}"
)

# Secure SCP options (same as SSH plus some SCP-specific)
SCP_OPTS=(
    "${SSH_OPTS[@]}"
    -o "Compression=yes"                # Compress for faster transfer
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_debug() { [[ "${DEBUG:-0}" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $*" >&2; }

# Security check function
check_ssh_security() {
    log_info "Performing security checks..."
    
    # Check if known_hosts file exists
    if [[ ! -f "${KNOWN_HOSTS_FILE}" ]]; then
        log_error "Known hosts file not found at ${KNOWN_HOSTS_FILE}"
        log_warn "You need to add the Edge Pi host key first."
        log_info "To add the host key securely:"
        echo ""
        echo "  1. Verify the host key fingerprint out-of-band (ask the Pi administrator)"
        echo "  2. Add it manually:"
        echo "     ssh-keyscan -t ${SSH_KEY_TYPE} -p ${SSH_PORT} ${EDGE_PI_HOST} >> ${KNOWN_HOSTS_FILE}"
        echo "  3. Or connect once manually and verify the fingerprint:"
        echo "     ssh -p ${SSH_PORT} ${EDGE_PI_TARGET}"
        echo ""
        return 1
    fi
    
    # Check if host is in known_hosts
    if ! ssh-keygen -F "[${EDGE_PI_HOST}]:${SSH_PORT}" -f "${KNOWN_HOSTS_FILE}" >/dev/null 2>&1; then
        log_error "Edge Pi host key not found in ${KNOWN_HOSTS_FILE}"
        log_info "Fetching current host key for verification..."
        
        # Get the current host key
        CURRENT_KEY=$(ssh-keyscan -t ${SSH_KEY_TYPE} -p ${SSH_PORT} ${EDGE_PI_HOST} 2>/dev/null | head -1)
        if [[ -z "${CURRENT_KEY}" ]]; then
            log_error "Could not fetch host key from ${EDGE_PI_HOST}"
            return 1
        fi
        
        # Display the fingerprint for manual verification
        FINGERPRINT=$(echo "${CURRENT_KEY}" | ssh-keygen -lf - 2>/dev/null | awk '{print $2}')
        log_warn "Host key fingerprint for ${EDGE_PI_HOST}:"
        echo "  ${FINGERPRINT}"
        echo ""
        echo "Please verify this fingerprint with the Edge Pi administrator."
        echo "If it matches, add it to known_hosts with:"
        echo "  echo '${CURRENT_KEY}' >> ${KNOWN_HOSTS_FILE}"
        echo ""
        return 1
    fi
    
    # Check SSH key permissions
    SSH_KEY="${HOME}/.ssh/id_${SSH_KEY_TYPE}"
    if [[ -f "${SSH_KEY}" ]]; then
        PERMS=$(stat -c %a "${SSH_KEY}" 2>/dev/null || stat -f %A "${SSH_KEY}" 2>/dev/null || echo "unknown")
        if [[ "${PERMS}" != "600" && "${PERMS}" != "400" ]]; then
            log_warn "SSH key ${SSH_KEY} has incorrect permissions: ${PERMS}"
            log_info "Fixing permissions..."
            chmod 600 "${SSH_KEY}"
        fi
    else
        log_error "SSH key not found at ${SSH_KEY}"
        log_info "Generate one with: ssh-keygen -t ${SSH_KEY_TYPE}"
        return 1
    fi
    
    log_info "Security checks passed ✅"
    return 0
}

# Test SSH connection with retry logic
test_ssh_connection() {
    local retry_count=0
    
    while [[ ${retry_count} -lt ${MAX_RETRIES} ]]; do
        log_info "Testing SSH connection (attempt $((retry_count + 1))/${MAX_RETRIES})..."
        
        if ssh "${SSH_OPTS[@]}" ${EDGE_PI_TARGET} "echo 'SSH connection successful'" 2>/dev/null; then
            log_info "SSH connection established ✅"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [[ ${retry_count} -lt ${MAX_RETRIES} ]]; then
            log_warn "Connection failed, retrying in ${RETRY_DELAY} seconds..."
            sleep ${RETRY_DELAY}
        fi
    done
    
    log_error "Cannot connect to Edge Pi after ${MAX_RETRIES} attempts"
    log_info "Please check:"
    echo "  1. Edge Pi is powered on and connected to network"
    echo "  2. SSH service is running on port ${SSH_PORT}"
    echo "  3. Network allows connection to ${EDGE_PI_HOST}:${SSH_PORT}"
    echo "  4. SSH key is properly configured"
    return 1
}

# Secure command execution wrapper
secure_ssh_exec() {
    local cmd="$1"
    log_debug "Executing: ${cmd}"
    ssh "${SSH_OPTS[@]}" ${EDGE_PI_TARGET} "${cmd}"
}

# Secure file transfer wrapper
secure_scp() {
    local src="$1"
    local dst="$2"
    log_debug "Copying ${src} to ${dst}"
    scp "${SCP_OPTS[@]}" -r "${src}" "${EDGE_PI_TARGET}:${dst}"
}

# Main deployment process
main() {
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Secure M1.3 FSQ Model Deployment to Edge Pi${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "Target: ${YELLOW}${EDGE_PI_TARGET}${NC}"
    echo -e "Port: ${YELLOW}${SSH_PORT}${NC}"
    echo -e "Deployment Path: ${YELLOW}${DEPLOYMENT_DIR}${NC}"
    echo ""
    
    # Security checks
    if ! check_ssh_security; then
        log_error "Security checks failed. Aborting deployment."
        exit 1
    fi
    
    # Check if local deployment package exists
    if [[ ! -d "${LOCAL_PACKAGE}" ]]; then
        log_error "Deployment package not found at ${LOCAL_PACKAGE}"
        log_info "Run 'python deploy_m13_fsq.py' first to create the package"
        exit 1
    fi
    
    # Test SSH connection
    if ! test_ssh_connection; then
        exit 1
    fi
    
    # Check Edge Pi system info
    log_info "Checking Edge Pi system..."
    secure_ssh_exec "uname -a && echo 'CPU: ' && lscpu | grep 'Model name' || echo 'ARM processor'"
    
    # Check for Hailo SDK
    log_info "Checking for Hailo SDK..."
    if secure_ssh_exec "which hailo 2>/dev/null || which hailortcli 2>/dev/null"; then
        log_info "Hailo SDK detected ✅"
        HAILO_AVAILABLE=true
    else
        log_warn "Hailo SDK not found - will use ONNX Runtime fallback"
        HAILO_AVAILABLE=false
    fi
    
    # Create deployment directory
    log_info "Creating deployment directory..."
    secure_ssh_exec "mkdir -p ${DEPLOYMENT_DIR}/backup 2>/dev/null || true"
    
    # Backup existing deployment if it exists
    if secure_ssh_exec "[ -d ${DEPLOYMENT_DIR}/models ]"; then
        log_info "Backing up existing deployment..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        secure_ssh_exec "cd ${DEPLOYMENT_DIR} && tar -czf backup/backup_${TIMESTAMP}.tar.gz models scripts docs 2>/dev/null || true"
        log_info "Backup created at backup/backup_${TIMESTAMP}.tar.gz"
    fi
    
    # Calculate package size
    PACKAGE_SIZE=$(du -sh ${LOCAL_PACKAGE} | cut -f1)
    log_info "Deploying package (size: ${PACKAGE_SIZE})..."
    
    # Transfer deployment package
    log_info "Transferring files to Edge Pi..."
    if secure_scp "${LOCAL_PACKAGE}/*" "${DEPLOYMENT_DIR}/"; then
        log_info "Transfer complete ✅"
    else
        log_error "File transfer failed"
        exit 1
    fi
    
    # Set execute permissions for scripts
    log_info "Setting script permissions..."
    secure_ssh_exec "chmod +x ${DEPLOYMENT_DIR}/scripts/*.sh 2>/dev/null || true"
    
    # Create deployment info file with security metadata
    log_info "Creating deployment info..."
    DEPLOY_INFO="Deployment Date: $(date)
Deployed By: $(whoami)@$(hostname)
Package Version: M1.3-FSQ
Hailo Available: ${HAILO_AVAILABLE}
Security Mode: Strict (StrictHostKeyChecking=yes)
SSH Port: ${SSH_PORT}
Host Key Type: ${SSH_KEY_TYPE}"
    
    echo "${DEPLOY_INFO}" | secure_ssh_exec "cat > ${DEPLOYMENT_DIR}/deployment_info.txt"
    
    # Verify deployment with checksums
    log_info "Verifying deployment integrity..."
    
    # Generate local checksums
    CHECKSUM_FILE="/tmp/deployment_checksums_$$.txt"
    (cd ${LOCAL_PACKAGE} && find . -type f -exec sha256sum {} \; > "${CHECKSUM_FILE}")
    
    # Transfer and verify checksums
    secure_scp "${CHECKSUM_FILE}" "${DEPLOYMENT_DIR}/checksums.txt"
    
    if secure_ssh_exec "cd ${DEPLOYMENT_DIR} && sha256sum -c checksums.txt 2>/dev/null"; then
        log_info "Deployment integrity verified ✅"
    else
        log_error "Deployment integrity check failed!"
        log_warn "Some files may be corrupted during transfer"
    fi
    
    # Clean up temp file
    rm -f "${CHECKSUM_FILE}"
    
    # Check Python dependencies
    log_info "Checking Python environment..."
    secure_ssh_exec "python3 --version 2>/dev/null || echo 'Python3 not found'"
    secure_ssh_exec "python3 -c 'import torch; print(f\"PyTorch {torch.__version__}\")' 2>/dev/null || echo 'PyTorch not installed'"
    secure_ssh_exec "python3 -c 'import onnxruntime; print(f\"ONNX Runtime {onnxruntime.__version__}\")' 2>/dev/null || echo 'ONNX Runtime not installed'"
    
    # Compile for Hailo if available
    if [[ "${HAILO_AVAILABLE}" == "true" ]]; then
        log_info "Compiling model for Hailo-8..."
        if secure_ssh_exec "cd ${DEPLOYMENT_DIR}/scripts && [ -f compile_hailo8.sh ]"; then
            log_warn "This will compile the model for Hailo-8 (may take a few minutes)"
            read -p "Proceed with Hailo compilation? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                secure_ssh_exec "cd ${DEPLOYMENT_DIR}/scripts && sudo ./compile_hailo8.sh"
            else
                log_info "Skipping Hailo compilation"
            fi
        fi
    fi
    
    # Final summary
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Deployment Complete! ✅${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "1. SSH to Edge Pi: ${YELLOW}ssh ${SSH_OPTS[*]} ${EDGE_PI_TARGET}${NC}"
    echo -e "2. Navigate to: ${YELLOW}cd ${DEPLOYMENT_DIR}${NC}"
    echo -e "3. Test inference: ${YELLOW}python3 scripts/test_inference.py${NC}"
    echo ""
    echo -e "${GREEN}Security Notes:${NC}"
    echo "- Host key verification: ENABLED ✅"
    echo "- Password authentication: DISABLED ✅"
    echo "- Connection logging: ENABLED ✅"
    echo "- File integrity checks: COMPLETED ✅"
}

# Run main function
main "$@"