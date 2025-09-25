#!/bin/bash

# Setup secure SSH connection to Edge Pi
# This script helps establish a secure SSH connection with proper host key verification

set -euo pipefail

# Configuration
EDGE_PI_HOST="${1:-100.127.242.78}"
EDGE_PI_USER="${2:-pi}"
SSH_PORT="${3:-22}"
SSH_KEY_TYPE="ed25519"
KNOWN_HOSTS_FILE="${HOME}/.ssh/known_hosts"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Secure SSH Setup for Edge Pi${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Target: ${YELLOW}${EDGE_PI_USER}@${EDGE_PI_HOST}:${SSH_PORT}${NC}"
echo ""

# Function to display host key info
display_host_key_info() {
    local host="$1"
    local port="$2"
    
    echo -e "${YELLOW}Fetching host key information...${NC}"
    
    # Get all host keys
    local keys=$(ssh-keyscan -p ${port} ${host} 2>/dev/null)
    
    if [[ -z "${keys}" ]]; then
        echo -e "${RED}Could not fetch host keys from ${host}:${port}${NC}"
        return 1
    fi
    
    echo ""
    echo -e "${BLUE}Host Key Fingerprints:${NC}"
    echo "----------------------------------------"
    
    # Display fingerprints for each key type
    for type in rsa ed25519 ecdsa; do
        local key=$(echo "${keys}" | grep "ssh-${type}" | head -1)
        if [[ ! -z "${key}" ]]; then
            local fp=$(echo "${key}" | ssh-keygen -lf - 2>/dev/null | awk '{print $2}')
            local fp_sha256=$(echo "${key}" | ssh-keygen -lf - -E sha256 2>/dev/null | awk '{print $2}')
            echo -e "${type^^}:"
            echo "  MD5:    ${fp}"
            echo "  SHA256: ${fp_sha256}"
        fi
    done
    
    echo "----------------------------------------"
    echo ""
    return 0
}

# Step 1: Check SSH directory
echo -e "${YELLOW}Step 1: Checking SSH configuration...${NC}"

if [[ ! -d "${HOME}/.ssh" ]]; then
    echo "Creating SSH directory..."
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
fi

# Step 2: Check for SSH keys
echo -e "${YELLOW}Step 2: Checking SSH keys...${NC}"

SSH_KEY="${HOME}/.ssh/id_${SSH_KEY_TYPE}"
if [[ ! -f "${SSH_KEY}" ]]; then
    echo -e "${YELLOW}No ${SSH_KEY_TYPE} key found.${NC}"
    read -p "Generate a new SSH key? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh-keygen -t ${SSH_KEY_TYPE} -f "${SSH_KEY}" -C "$(whoami)@$(hostname)-to-edgepi"
        echo -e "${GREEN}SSH key generated at ${SSH_KEY}${NC}"
    else
        echo -e "${RED}SSH key required for secure connection. Exiting.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}SSH key found at ${SSH_KEY}${NC}"
fi

# Check key permissions
PERMS=$(stat -c %a "${SSH_KEY}" 2>/dev/null || stat -f %A "${SSH_KEY}" 2>/dev/null)
if [[ "${PERMS}" != "600" && "${PERMS}" != "400" ]]; then
    echo "Fixing SSH key permissions..."
    chmod 600 "${SSH_KEY}"
    echo -e "${GREEN}Permissions fixed${NC}"
fi

# Step 3: Display public key for copying to Edge Pi
echo ""
echo -e "${YELLOW}Step 3: Public key for Edge Pi${NC}"
echo "----------------------------------------"
echo -e "${BLUE}Copy this public key to the Edge Pi's ~/.ssh/authorized_keys:${NC}"
echo ""
cat "${SSH_KEY}.pub"
echo ""
echo "----------------------------------------"
echo ""
echo "To add this key to the Edge Pi, run on the Pi:"
echo -e "${YELLOW}echo '$(cat ${SSH_KEY}.pub)' >> ~/.ssh/authorized_keys${NC}"
echo ""
read -p "Press Enter when the public key has been added to the Edge Pi..."

# Step 4: Get and verify host key
echo ""
echo -e "${YELLOW}Step 4: Host key verification${NC}"

# Check if host is already in known_hosts
if ssh-keygen -F "[${EDGE_PI_HOST}]:${SSH_PORT}" -f "${KNOWN_HOSTS_FILE}" >/dev/null 2>&1; then
    echo -e "${GREEN}Host already in known_hosts file${NC}"
    echo "Current host key fingerprint:"
    ssh-keygen -F "[${EDGE_PI_HOST}]:${SSH_PORT}" -f "${KNOWN_HOSTS_FILE}" | \
        ssh-keygen -lf - 2>/dev/null | awk '{print $2}'
    
    read -p "Do you want to update the host key? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove old key
        ssh-keygen -R "[${EDGE_PI_HOST}]:${SSH_PORT}" -f "${KNOWN_HOSTS_FILE}" 2>/dev/null
        echo "Old host key removed"
    else
        echo "Keeping existing host key"
        SKIP_HOST_KEY=true
    fi
fi

if [[ "${SKIP_HOST_KEY:-false}" != "true" ]]; then
    # Display host key information
    if ! display_host_key_info "${EDGE_PI_HOST}" "${SSH_PORT}"; then
        exit 1
    fi
    
    echo -e "${RED}IMPORTANT SECURITY NOTICE:${NC}"
    echo "========================================="
    echo "You MUST verify these fingerprints match the actual Edge Pi host keys!"
    echo ""
    echo "Verification methods:"
    echo "1. Ask the Edge Pi administrator for the fingerprints"
    echo "2. If you have physical access, run on the Pi:"
    echo "   ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub"
    echo "3. Compare with fingerprints from a trusted previous connection"
    echo "========================================="
    echo ""
    
    read -p "Do these fingerprints match the Edge Pi? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Adding host key to known_hosts..."
        ssh-keyscan -t ${SSH_KEY_TYPE} -p ${SSH_PORT} ${EDGE_PI_HOST} >> "${KNOWN_HOSTS_FILE}" 2>/dev/null
        echo -e "${GREEN}Host key added successfully${NC}"
    else
        echo -e "${RED}Host key verification failed. Connection not established.${NC}"
        echo "For security, we cannot proceed without verifying the host key."
        exit 1
    fi
fi

# Step 5: Test secure connection
echo ""
echo -e "${YELLOW}Step 5: Testing secure connection...${NC}"

SSH_OPTS=(
    -o "StrictHostKeyChecking=yes"
    -o "PasswordAuthentication=no"
    -o "PubkeyAuthentication=yes"
    -o "ConnectTimeout=10"
    -p "${SSH_PORT}"
)

if ssh "${SSH_OPTS[@]}" "${EDGE_PI_USER}@${EDGE_PI_HOST}" "echo 'Secure connection successful!'" 2>/dev/null; then
    echo -e "${GREEN}✅ Secure SSH connection established!${NC}"
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "You can now use the secure deployment script:"
    echo -e "${YELLOW}./deploy_to_edge_pi_secure.sh${NC}"
    echo ""
    echo "Security features enabled:"
    echo "  ✅ Host key verification (StrictHostKeyChecking=yes)"
    echo "  ✅ Key-based authentication only"
    echo "  ✅ No password authentication"
    echo "  ✅ Known hosts verification"
else
    echo -e "${RED}Connection test failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Verify the public key was added to Edge Pi's authorized_keys"
    echo "2. Check that SSH service is running on the Edge Pi"
    echo "3. Verify network connectivity to ${EDGE_PI_HOST}:${SSH_PORT}"
    echo "4. Check Edge Pi's SSH configuration allows key authentication"
    exit 1
fi

# Step 6: Create SSH config entry for convenience
echo ""
read -p "Create SSH config entry for easy access? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    SSH_CONFIG="${HOME}/.ssh/config"
    CONFIG_ENTRY="
# Edge Pi Secure Connection
Host edgepi
    HostName ${EDGE_PI_HOST}
    Port ${SSH_PORT}
    User ${EDGE_PI_USER}
    IdentityFile ${SSH_KEY}
    StrictHostKeyChecking yes
    PasswordAuthentication no
    PubkeyAuthentication yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
"
    
    echo "${CONFIG_ENTRY}" >> "${SSH_CONFIG}"
    chmod 600 "${SSH_CONFIG}"
    
    echo -e "${GREEN}SSH config entry added${NC}"
    echo "You can now connect with: ${YELLOW}ssh edgepi${NC}"
fi

echo ""
echo -e "${BLUE}Security Best Practices:${NC}"
echo "1. Always verify host key fingerprints out-of-band"
echo "2. Use ed25519 keys (most secure and efficient)"
echo "3. Keep StrictHostKeyChecking=yes in production"
echo "4. Regularly rotate SSH keys (every 6-12 months)"
echo "5. Monitor SSH logs for unauthorized access attempts"
echo ""