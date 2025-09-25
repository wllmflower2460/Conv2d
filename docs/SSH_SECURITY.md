# SSH Security Best Practices for Edge Pi Deployment

## Overview
This document outlines the security measures implemented for SSH connections to Edge Pi devices, addressing the critical vulnerability of missing host key verification.

## Security Issue Addressed
**Problem**: The original `deploy_to_edge_pi.sh` script did not verify SSH host key authenticity, making it vulnerable to man-in-the-middle (MITM) attacks.

**Solution**: Implemented strict host key checking with `StrictHostKeyChecking=yes` and proper known_hosts management.

## Security Features

### 1. Host Key Verification
```bash
# Always verify host keys
-o "StrictHostKeyChecking=yes"
-o "UserKnownHostsFile=${HOME}/.ssh/known_hosts"
```

**Why it matters**: Without host key verification, an attacker could intercept the connection and steal credentials or inject malicious code.

### 2. Key-Based Authentication Only
```bash
# Disable password authentication
-o "PasswordAuthentication=no"
-o "PubkeyAuthentication=yes"
```

**Why it matters**: Passwords can be brute-forced or intercepted. SSH keys with proper permissions are much more secure.

### 3. Connection Security Options
```bash
SSH_OPTS=(
    -o "StrictHostKeyChecking=yes"      # Verify host key
    -o "PasswordAuthentication=no"       # No passwords
    -o "PubkeyAuthentication=yes"       # Keys only
    -o "ConnectTimeout=10"              # Timeout for hanging connections
    -o "ServerAliveInterval=60"         # Detect broken connections
    -o "BatchMode=yes"                  # Fail instead of prompting
    -o "LogLevel=VERBOSE"               # Security audit trail
)
```

## Initial Setup Process

### Step 1: Run Setup Script
```bash
./scripts/setup_secure_ssh.sh [host] [user] [port]
# Example: ./scripts/setup_secure_ssh.sh 100.127.242.78 pi 22
```

### Step 2: Verify Host Key Fingerprint
The script will display the host's key fingerprint. You MUST verify this through a secure channel:

#### Option A: Physical Access
If you have physical access to the Edge Pi:
```bash
# On the Edge Pi
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

#### Option B: Trusted Administrator
Contact the Edge Pi administrator and verify the fingerprint over a trusted channel (not email).

#### Option C: Certificate Authority
For production deployments, consider using SSH Certificate Authority for automated trust.

### Step 3: Add Public Key to Edge Pi
Copy your public key to the Edge Pi's authorized_keys:
```bash
# On your machine
cat ~/.ssh/id_ed25519.pub

# On the Edge Pi
echo "your-public-key-here" >> ~/.ssh/authorized_keys
```

## Deployment with Security

### Using Secure Deployment Script
```bash
# After setup, use the secure script
./deploy_to_edge_pi_secure.sh
```

Features:
- ✅ Host key verification
- ✅ Integrity checks (SHA256 checksums)
- ✅ Secure transfer with compression
- ✅ Audit logging
- ✅ Automatic retries with exponential backoff

### Manual Secure Connection
```bash
ssh -o StrictHostKeyChecking=yes \
    -o PasswordAuthentication=no \
    -o PubkeyAuthentication=yes \
    pi@100.127.242.78
```

## Security Hardening Checklist

### Local Machine
- [ ] SSH keys have correct permissions (600 or 400)
- [ ] Known_hosts file is maintained and verified
- [ ] No plaintext passwords in scripts
- [ ] Regular key rotation (every 6-12 months)

### Edge Pi Configuration
- [ ] SSH daemon configured for key-only authentication
- [ ] Root login disabled
- [ ] SSH on non-standard port (optional)
- [ ] Fail2ban or similar brute-force protection
- [ ] Regular security updates

### Network Security
- [ ] Use VPN for additional security layer
- [ ] Implement IP whitelisting if possible
- [ ] Monitor SSH logs for anomalies
- [ ] Use jump hosts for production environments

## Common Security Mistakes to Avoid

### ❌ Never Do This:
```bash
# INSECURE - Disables host checking
ssh -o StrictHostKeyChecking=no user@host

# INSECURE - Accepts any host key
ssh -o UserKnownHostsFile=/dev/null user@host

# INSECURE - Uses password in command
sshpass -p 'password' ssh user@host
```

### ✅ Always Do This:
```bash
# SECURE - Verifies host and uses keys
ssh -o StrictHostKeyChecking=yes \
    -o PasswordAuthentication=no \
    user@host
```

## Troubleshooting

### Host Key Changed Warning
If you see:
```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

**DO NOT** blindly remove the old key. This could indicate:
1. The Edge Pi was reinstalled (verify with administrator)
2. Network configuration changed
3. **Potential MITM attack**

Verify the new key through a secure channel before proceeding.

### Permission Denied
Check:
1. Public key is in Edge Pi's authorized_keys
2. SSH key permissions are 600 or 400
3. Edge Pi allows key authentication
4. Correct username and key type

### Connection Timeout
1. Verify network connectivity
2. Check if SSH port is correct
3. Ensure Edge Pi is powered on
4. Check firewall rules

## Audit and Monitoring

### Enable SSH Logging
```bash
# In deployment script
-o "LogLevel=VERBOSE"
```

### Monitor Access
```bash
# On Edge Pi - Check auth logs
sudo tail -f /var/log/auth.log

# Failed login attempts
sudo grep "Failed password" /var/log/auth.log
```

### Regular Security Audits
1. Review authorized_keys files
2. Check for unauthorized SSH keys
3. Verify known_hosts entries
4. Rotate keys periodically

## Advanced Security Options

### SSH Certificate Authority
For multiple Edge Pi deployments, consider implementing SSH CA:
```bash
# Generate CA key
ssh-keygen -t ed25519 -f edge_pi_ca

# Sign host keys
ssh-keygen -s edge_pi_ca -I "edge-pi-01" -h /etc/ssh/ssh_host_ed25519_key.pub

# Sign user keys
ssh-keygen -s edge_pi_ca -I "developer" ~/.ssh/id_ed25519.pub
```

### Hardware Security Keys
Support FIDO2/U2F hardware keys for additional security:
```bash
# Generate key backed by hardware token
ssh-keygen -t ed25519-sk
```

### Bastion/Jump Host
For production environments:
```bash
# Connect through bastion
ssh -J bastion.example.com pi@edge-pi-internal
```

## References
- [OpenSSH Security Best Practices](https://www.ssh.com/academy/ssh/security)
- [NIST Guidelines for SSH](https://nvlpubs.nist.gov/nistpubs/ir/2015/NIST.IR.7966.pdf)
- [SSH Hardening Guide](https://www.sshaudit.com/)
- [Mozilla SSH Guidelines](https://infosec.mozilla.org/guidelines/openssh)

---

*Last Updated: 2025-09-24*
*Security is not optional - it's mandatory for production deployments*