#!/bin/bash
# GPUSRV Setup Script for CVAT + Firebase Integration
# Run this on your GPUSRV to prepare for image annotation workflow

set -e

echo "🚀 Setting up GPUSRV for Firebase → CVAT annotation pipeline..."

# Create necessary directories
echo "📁 Creating annotation directories..."
sudo mkdir -p /data/cvat-annotations
sudo mkdir -p /data/cvat-exports
sudo mkdir -p /opt/firebase-tools
sudo chown -R $USER:$USER /data/cvat-annotations /data/cvat-exports

# Install Python dependencies for Firebase transfer
echo "🐍 Installing Python dependencies..."
pip3 install firebase-admin requests python-dotenv tqdm

# Create Firebase service account directory
echo "🔑 Setting up Firebase credentials directory..."
mkdir -p ~/.firebase
chmod 700 ~/.firebase

echo """
📋 Next manual steps required:

1. **Firebase Service Account Setup:**
   - Go to Firebase Console → Project Settings → Service Accounts
   - Generate new private key
   - Copy JSON file to GPUSRV: ~/.firebase/service-account.json
   - Update bucket name in transfer script

2. **CVAT Setup (if not already done):**
   docker run -dit --name cvat_server --restart=always \\
     -p 8080:8080 \\
     -v /data/cvat-data:/home/django/data \\
     -v /data/cvat-annotations:/home/django/keys \\
     -v /data/cvat-exports:/home/django/models \\
     cvat/server

3. **Transfer Script Usage:**
   # Copy transfer script to GPUSRV
   scp firebase_to_cvat_transfer.py gpusrv.tailfdc654.ts.net:~/

   # Run transfer (on GPUSRV)
   cd ~ && python3 firebase_to_cvat_transfer.py \\
     --service-account ~/.firebase/service-account.json \\
     --limit 25 \\
     --priority 7 \\
     --output-dir /data/cvat-annotations

4. **CVAT Project Creation:**
   - Access CVAT at http://gpusrv.tailfdc654.ts.net:8080
   - Create new project: 'DataDogs Pose Estimation'
   - Import images from /data/cvat-annotations
   - Set up keypoint annotation labels for dog joints

5. **Annotation Workflow:**
   - Annotate uncertain poses with corrected keypoints
   - Export annotations in CVAT format
   - Use return script to update Firebase with corrections

🔗 **Tailscale Access:**
   Your GPUSRV is accessible at: gpusrv.tailfdc654.ts.net
   CVAT will be at: http://gpusrv.tailfdc654.ts.net:8080
"""

echo "✅ GPUSRV setup complete!"