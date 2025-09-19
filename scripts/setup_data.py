#!/usr/bin/env python3
"""
Setup script for downloading and preparing the pneumonia dataset
"""
import os
import subprocess
import zipfile
from pathlib import Path
import argparse


def download_dataset(data_dir: str = "data"):
    """Download the pneumonia dataset from Kaggle"""
    print("Downloading pneumonia dataset from Kaggle...")
    
    # Check if kaggle CLI is installed
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Kaggle CLI not found. Please install with: pip install kaggle")
        print("And configure your API credentials: https://github.com/Kaggle/kaggle-api")
        return False
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Download dataset
    try:
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "paultimothymooney/chest-xray-pneumonia",
            "-p", str(data_path)
        ], check=True)
        
        # Extract dataset
        zip_path = data_path / "chest-xray-pneumonia.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            
            # Remove zip file
            zip_path.unlink()
            print(f"Dataset extracted to {data_path}")
            return True
        else:
            print("ERROR: Downloaded file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR downloading dataset: {e}")
        return False


def verify_dataset(data_dir: str = "data/chest_xray"):
    """Verify dataset structure and count files"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"ERROR: Dataset directory {data_path} does not exist")
        return False
    
    expected_structure = {
        'train': ['NORMAL', 'PNEUMONIA'],
        'val': ['NORMAL', 'PNEUMONIA'],
        'test': ['NORMAL', 'PNEUMONIA']
    }
    
    print("Verifying dataset structure...")
    total_files = 0
    
    for split, classes in expected_structure.items():
        split_path = data_path / split
        if not split_path.exists():
            print(f"ERROR: Missing {split} directory")
            return False
            
        print(f"\n{split.upper()} split:")
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"  ERROR: Missing {class_name} directory")
                return False
            
            # Count files
            files = list(class_path.glob("*.jpeg")) + list(class_path.glob("*.jpg"))
            count = len(files)
            total_files += count
            print(f"  {class_name}: {count} images")
    
    print(f"\nTotal images: {total_files}")
    print("Dataset structure verified successfully!")
    return True


def create_sample_dataset(data_dir: str = "data", sample_size: int = 100):
    """Create a smaller sample dataset for testing"""
    import shutil
    import random
    
    source_path = Path(data_dir) / "chest_xray"
    sample_path = Path(data_dir) / "chest_xray_sample"
    
    if not source_path.exists():
        print("ERROR: Source dataset not found. Please download first.")
        return False
    
    print(f"Creating sample dataset with {sample_size} images per class...")
    
    # Remove existing sample dataset
    if sample_path.exists():
        shutil.rmtree(sample_path)
    
    # Create sample structure
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (sample_path / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Get source files
            source_class_path = source_path / split / class_name
            source_files = list(source_class_path.glob("*.jpeg")) + list(source_class_path.glob("*.jpg"))
            
            # Sample files
            if split == 'train':
                sample_count = sample_size
            elif split == 'val':
                sample_count = sample_size // 5
            else:  # test
                sample_count = sample_size // 10
                
            sample_files = random.sample(source_files, min(sample_count, len(source_files)))
            
            # Copy files
            for i, source_file in enumerate(sample_files):
                dest_file = sample_path / split / class_name / f"{class_name.lower()}_{i:04d}.jpg"
                shutil.copy2(source_file, dest_file)
            
            print(f"  {split}/{class_name}: {len(sample_files)} images")
    
    print(f"Sample dataset created at {sample_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup pneumonia detection dataset")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--verify", action="store_true", help="Verify dataset structure")
    parser.add_argument("--sample", type=int, help="Create sample dataset with N images per class")
    
    args = parser.parse_args()
    
    if args.download:
        success = download_dataset(args.data_dir)
        if not success:
            return 1
    
    if args.verify:
        success = verify_dataset(os.path.join(args.data_dir, "chest_xray"))
        if not success:
            return 1
    
    if args.sample:
        success = create_sample_dataset(args.data_dir, args.sample)
        if not success:
            return 1
    
    if not any([args.download, args.verify, args.sample]):
        # Default: download and verify
        print("No action specified. Downloading and verifying dataset...")
        if download_dataset(args.data_dir):
            verify_dataset(os.path.join(args.data_dir, "chest_xray"))
    
    return 0


if __name__ == "__main__":
    exit(main())