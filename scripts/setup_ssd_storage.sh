#!/bin/bash
# Setup script for SSD storage and dataset directories

echo "================================================"
echo "Conv2d M1.6 SSD Storage Setup"
echo "================================================"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
   echo "Please run with sudo: sudo bash setup_ssd_storage.sh"
   exit 1
fi

# Mount the SSD if not already mounted
echo "Checking SSD mount status..."
if ! mountpoint -q /mnt/ssd; then
    echo "Creating mount point /mnt/ssd..."
    mkdir -p /mnt/ssd
    
    echo "Mounting /dev/sdd3 to /mnt/ssd..."
    mount /dev/sdd3 /mnt/ssd
    
    # Make it persistent (optional)
    echo "Adding to /etc/fstab for persistent mount..."
    echo "# Conv2d datasets SSD" >> /etc/fstab
    echo "/dev/sdd3 /mnt/ssd ext4 defaults 0 2" >> /etc/fstab
else
    echo "SSD already mounted at /mnt/ssd"
fi

# Check available space
echo ""
echo "Storage Status:"
df -h /mnt/ssd

# Create directory structure
echo ""
echo "Creating dataset directory structure..."

# Primary SSD directories
mkdir -p /mnt/ssd/Conv2d_Datasets/{synthetic,semi_synthetic,real_quadruped,animal_locomotion,har_adapted,dynamic_validation,processed}

# Semi-synthetic subdirs
mkdir -p /mnt/ssd/Conv2d_Datasets/semi_synthetic/tartanvo

# Real quadruped subdirs
mkdir -p /mnt/ssd/Conv2d_Datasets/real_quadruped/{legkilo,cear_mini_cheetah,spot_assessment}

# Animal locomotion subdirs
mkdir -p /mnt/ssd/Conv2d_Datasets/animal_locomotion/{horse_gaits,dog_behavior}

# HAR adapted subdirs
mkdir -p /mnt/ssd/Conv2d_Datasets/har_adapted/{pamap2,opportunity}

# Dynamic validation subdirs
mkdir -p /mnt/ssd/Conv2d_Datasets/dynamic_validation/{uzh_fpv,blackbird}

# Processed data subdirs
mkdir -p /mnt/ssd/Conv2d_Datasets/processed/{tartanvo,legkilo,pamap2,horse_gaits,dog_behavior}

# Create backup structure on RAID
echo "Creating backup structure on RAID..."
mkdir -p /mnt/raid1/Conv2d_Datasets_Backup

# Set permissions
echo "Setting permissions..."
chown -R $SUDO_USER:$SUDO_USER /mnt/ssd/Conv2d_Datasets
chown -R $SUDO_USER:$SUDO_USER /mnt/raid1/Conv2d_Datasets_Backup

# Create symlink in project directory for easy access
echo "Creating symlink in project directory..."
if [ ! -L "/home/$SUDO_USER/Development/Conv2d/datasets" ]; then
    ln -s /mnt/ssd/Conv2d_Datasets /home/$SUDO_USER/Development/Conv2d/datasets
    chown -h $SUDO_USER:$SUDO_USER /home/$SUDO_USER/Development/Conv2d/datasets
fi

echo ""
echo "================================================"
echo "✅ Storage setup complete!"
echo "================================================"
echo ""
echo "Directory structure created:"
echo "  Primary: /mnt/ssd/Conv2d_Datasets/"
echo "  Backup:  /mnt/raid1/Conv2d_Datasets_Backup/"
echo "  Symlink: ~/Development/Conv2d/datasets/"
echo ""
echo "Available space on SSD:"
df -h /mnt/ssd | grep -v Filesystem
echo ""
echo "To download first dataset, run:"
echo "  cd ~/Development/Conv2d"
echo "  python scripts/download_tartanvo.py --env Downtown --process"
echo ""