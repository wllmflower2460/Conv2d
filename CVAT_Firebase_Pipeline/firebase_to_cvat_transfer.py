#!/usr/bin/env python3
"""
Firebase to CVAT Image Transfer Pipeline
Transfers uncertain pose images from Firebase Storage to GPUSRV for CVAT annotation

Usage:
  python firebase_to_cvat_transfer.py --limit 50 --priority 7
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import asyncio
import requests
from typing import List, Dict, Optional

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
except ImportError:
    print("❌ Firebase Admin SDK not installed. Install with:")
    print("pip install firebase-admin")
    sys.exit(1)

class FirebaseToCVATTransfer:
    def __init__(self, service_account_path: str, gpusrv_host: str):
        """Initialize Firebase connection and GPUSRV settings"""
        
        # Initialize Firebase Admin SDK
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'your-project-id.appspot.com'  # Replace with your bucket
            })
        
        self.db = firestore.client()
        self.bucket = storage.bucket()
        self.gpusrv_host = gpusrv_host
        self.download_dir = Path("/tmp/firebase_images")
        self.download_dir.mkdir(exist_ok=True)
        
    async def get_uncertain_poses(self, limit: int = 50, min_priority: int = 5) -> List[Dict]:
        """Fetch uncertain pose data from Firestore"""
        print(f"🔍 Fetching uncertain poses (limit={limit}, priority>={min_priority})")
        
        # Query uncertain poses collection
        poses_ref = self.db.collection('uncertain_poses')
        query = (poses_ref
                .where('status', '==', 'pending')
                .where('priority', '>=', min_priority)
                .order_by('priority', direction=firestore.Query.DESCENDING)
                .order_by('timestamp', direction=firestore.Query.ASCENDING)
                .limit(limit))
        
        docs = query.stream()
        uncertain_poses = []
        
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            uncertain_poses.append(data)
        
        print(f"📊 Found {len(uncertain_poses)} uncertain poses")
        return uncertain_poses
    
    async def download_training_images(self, pose_data_list: List[Dict]) -> List[Dict]:
        """Download images from Firebase Storage"""
        print("📥 Downloading training images from Firebase Storage...")
        
        successful_downloads = []
        
        for pose_data in pose_data_list:
            try:
                pose_id = pose_data['id']
                image_path = f"training_images/{pose_id}.jpg"
                
                # Download from Firebase Storage
                blob = self.bucket.blob(image_path)
                local_path = self.download_dir / f"{pose_id}.jpg"
                
                if blob.exists():
                    blob.download_to_filename(str(local_path))
                    
                    # Add local path to pose data
                    pose_data['local_image_path'] = str(local_path)
                    pose_data['image_filename'] = f"{pose_id}.jpg"
                    successful_downloads.append(pose_data)
                    
                    print(f"✅ Downloaded: {pose_id}.jpg")
                else:
                    print(f"❌ Image not found in storage: {image_path}")
                    
            except Exception as e:
                print(f"❌ Failed to download {pose_data.get('id', 'unknown')}: {e}")
        
        print(f"📊 Successfully downloaded {len(successful_downloads)}/{len(pose_data_list)} images")
        return successful_downloads
    
    async def create_cvat_dataset(self, pose_data_list: List[Dict], dataset_name: str) -> Optional[Dict]:
        """Create CVAT dataset and upload images"""
        print(f"🎯 Creating CVAT dataset: {dataset_name}")
        
        # CVAT API endpoints (adjust for your CVAT setup)
        cvat_api_base = f"http://{self.gpusrv_host}:8080/api/v1"
        
        # Create project
        project_data = {
            "name": dataset_name,
            "labels": [
                {
                    "name": "dog_keypoints",
                    "attributes": [
                        {"name": "confidence", "mutable": True, "input_type": "number"},
                        {"name": "joint_group", "mutable": True, "input_type": "select", 
                         "values": ["head", "forelegs", "hindlegs", "trunk", "tail", "all"]},
                        {"name": "priority", "mutable": True, "input_type": "number"}
                    ]
                }
            ]
        }
        
        try:
            # This is a template - you'll need to adapt based on your CVAT API authentication
            headers = {
                'Authorization': 'Token your-cvat-api-token',  # Replace with actual token
                'Content-Type': 'application/json'
            }
            
            response = requests.post(f"{cvat_api_base}/projects", 
                                   json=project_data, headers=headers)
            
            if response.status_code == 201:
                project = response.json()
                print(f"✅ Created CVAT project: {project['id']}")
                
                # Create task within project
                task_data = {
                    "name": f"{dataset_name}_task",
                    "project_id": project['id'],
                    "labels": project_data['labels']
                }
                
                # Upload images to task (this requires multipart form data)
                # Implementation depends on your CVAT setup
                
                return {"project_id": project['id'], "pose_data": pose_data_list}
            else:
                print(f"❌ Failed to create CVAT project: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ CVAT API error: {e}")
            return None
    
    async def transfer_to_gpusrv(self, pose_data_list: List[Dict], output_dir: str = "/data/cvat-annotations"):
        """Transfer images and metadata to GPUSRV file system"""
        print(f"🚀 Transferring {len(pose_data_list)} images to GPUSRV...")
        
        # Create transfer manifest
        manifest = {
            "transfer_timestamp": datetime.now().isoformat(),
            "total_images": len(pose_data_list),
            "images": []
        }
        
        for pose_data in pose_data_list:
            if 'local_image_path' not in pose_data:
                continue
                
            try:
                # Copy to GPUSRV directory (adjust path as needed)
                gpusrv_path = f"{output_dir}/{pose_data['image_filename']}"
                
                # Use scp or rsync for remote transfer
                # For local testing, just copy files
                import shutil
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(pose_data['local_image_path'], gpusrv_path)
                
                # Add to manifest
                image_metadata = {
                    "filename": pose_data['image_filename'],
                    "firebase_id": pose_data['id'],
                    "confidence": pose_data.get('confidence', 0.0),
                    "priority": pose_data.get('priority', 1),
                    "joint_group": pose_data.get('jointGroup', 'all'),
                    "timestamp": pose_data.get('timestamp', ''),
                    "device_id": pose_data.get('deviceId', ''),
                    "detected_poses": pose_data.get('detectedPoses', []),
                    "local_path": gpusrv_path
                }
                manifest["images"].append(image_metadata)
                
                print(f"✅ Transferred: {pose_data['image_filename']}")
                
            except Exception as e:
                print(f"❌ Failed to transfer {pose_data['image_filename']}: {e}")
        
        # Save manifest
        manifest_path = f"{output_dir}/transfer_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"📊 Transfer complete: {len(manifest['images'])} images")
        print(f"📝 Manifest saved to: {manifest_path}")
        
        return manifest
    
    async def update_firebase_status(self, pose_ids: List[str], status: str = "in_progress"):
        """Update Firebase status to indicate images are being annotated"""
        print(f"🔄 Updating {len(pose_ids)} pose records to status: {status}")
        
        batch = self.db.batch()
        
        for pose_id in pose_ids:
            doc_ref = self.db.collection('uncertain_poses').document(pose_id)
            batch.update(doc_ref, {
                'status': status,
                'annotation_started': firestore.SERVER_TIMESTAMP,
                'annotation_platform': 'CVAT',
                'gpusrv_transfer': True
            })
        
        batch.commit()
        print("✅ Firebase status updated")

async def main():
    parser = argparse.ArgumentParser(description='Transfer Firebase training images to CVAT')
    parser.add_argument('--service-account', required=True, 
                       help='Path to Firebase service account JSON file')
    parser.add_argument('--gpusrv-host', default='gpusrv.tailfdc654.ts.net',
                       help='GPUSRV hostname or IP')
    parser.add_argument('--limit', type=int, default=50,
                       help='Maximum number of images to transfer')
    parser.add_argument('--priority', type=int, default=5,
                       help='Minimum priority threshold (1-10)')
    parser.add_argument('--output-dir', default='/data/cvat-annotations',
                       help='Output directory on GPUSRV')
    parser.add_argument('--dataset-name', 
                       default=f"pose_estimation_{datetime.now().strftime('%Y%m%d_%H%M')}",
                       help='CVAT dataset name')
    
    args = parser.parse_args()
    
    # Initialize transfer pipeline
    transfer = FirebaseToCVATTransfer(args.service_account, args.gpusrv_host)
    
    try:
        # Step 1: Get uncertain poses from Firestore
        uncertain_poses = await transfer.get_uncertain_poses(args.limit, args.priority)
        
        if not uncertain_poses:
            print("❌ No uncertain poses found matching criteria")
            return
        
        # Step 2: Download training images
        pose_data_with_images = await transfer.download_training_images(uncertain_poses)
        
        if not pose_data_with_images:
            print("❌ No images downloaded successfully")
            return
        
        # Step 3: Transfer to GPUSRV
        manifest = await transfer.transfer_to_gpusrv(pose_data_with_images, args.output_dir)
        
        # Step 4: Update Firebase status
        pose_ids = [pose['id'] for pose in pose_data_with_images]
        await transfer.update_firebase_status(pose_ids, "in_progress")
        
        # Step 5: Optional - Create CVAT dataset
        # cvat_dataset = await transfer.create_cvat_dataset(pose_data_with_images, args.dataset_name)
        
        print(f"""
🎉 Transfer Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Images transferred: {len(manifest['images'])}
📁 Output directory: {args.output_dir}
📝 Manifest file: {args.output_dir}/transfer_manifest.json
🔄 Firebase status updated: in_progress

Next steps:
1. SSH to GPUSRV and verify images in {args.output_dir}
2. Import images into CVAT annotation project
3. Begin pose keypoint annotation
4. Export annotations and update Firebase when complete
        """)
        
    except Exception as e:
        print(f"❌ Transfer failed: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())