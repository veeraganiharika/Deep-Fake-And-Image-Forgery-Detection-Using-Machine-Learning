import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional

def setup_kaggle_credentials(username: str, key: str) -> None:
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    credentials = {
        "username": username,
        "key": key
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    print(f"✓ Kaggle credentials saved to {kaggle_json}")

def check_dataset_exists(dataset_path: str) -> bool:
    """Check if dataset is already downloaded and has images"""
    path = Path(dataset_path)
    
    if not path.exists():
        return False
    
    # Check if there are any image files in the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    for file in path.rglob('*'):
        if file.suffix.lower() in image_extensions:
            return True
    
    # Check for zip files that might contain the dataset
    for file in path.rglob('*.zip'):
        if file.exists():
            return True
    
    return False

def extract_all_zips(directory: Path) -> None:
    """Recursively extract all zip files in directory"""
    for zip_file in directory.rglob('*.zip'):
        print(f"  Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract to the same directory as the zip file
                extract_path = zip_file.parent
                zip_ref.extractall(extract_path)
            print(f"  ✓ Extracted to: {extract_path}")
            
            # Optionally remove the zip file after extraction
            # zip_file.unlink()
            
        except Exception as e:
            print(f"  ✗ Error extracting {zip_file.name}: {e}")

def download_dataset(dataset_name: str, download_path: str, force_download: bool = False) -> bool:
    """Download dataset from Kaggle"""
    path = Path(download_path)
    
    # Check if dataset already exists
    if not force_download and check_dataset_exists(download_path):
        print(f"✓ Dataset already exists at {download_path}")
        print("  Skipping download. Use force_download=True to re-download.")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset: {dataset_name}")
        path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset (unzip=True will extract it automatically)
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"✓ Dataset downloaded to {download_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def find_all_images(directory: Path) -> list:
    """Find all image files in directory and subdirectories"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    images = []
    
    for file in directory.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            images.append(file)
    
    return images

def organize_dataset(dataset_path: str) -> Optional[str]:
    """Organize dataset into train/test folders with authentic/forged subfolders"""
    print("\nOrganizing dataset...")
    base_path = Path(dataset_path)
    
    # First, extract any zip files
    print("Checking for zip files...")
    extract_all_zips(base_path)
    
    # Find all images
    all_images = find_all_images(base_path)
    print(f"Found {len(all_images)} total images")
    
    if len(all_images) == 0:
        print("✗ No images found in the dataset!")
        print("  Please check if the dataset was downloaded correctly.")
        return None
    
    # Create organized structure
    organized_path = base_path / 'organized'
    train_path = organized_path / 'train'
    test_path = organized_path / 'test'
    
    # Create directories
    for split in ['train', 'test']:
        split_path = organized_path / split
        (split_path / 'authentic').mkdir(parents=True, exist_ok=True)
        (split_path / 'forged').mkdir(parents=True, exist_ok=True)
    
    # Organize images based on their current location/naming
    # This is dataset-specific logic - adjust based on your dataset structure
    authentic_count = 0
    forged_count = 0
    train_count = 0
    test_count = 0
    
    for img_path in all_images:
        # Skip if already in organized folder
        if 'organized' in img_path.parts:
            continue
        
        # Determine if authentic or forged based on path or filename
        # Adjust these rules based on your dataset's naming convention
        path_str = str(img_path).lower()
        filename = img_path.name.lower()
        
        is_forged = any(keyword in path_str for keyword in 
                       ['forged', 'fake', 'tampered', 'manipulated', 'spliced'])
        is_authentic = any(keyword in path_str for keyword in 
                          ['authentic', 'real', 'original', 'pristine'])
        
        # If neither keyword found, try to infer from folder structure
        if not is_forged and not is_authentic:
            # Check parent folder names
            parent_parts = [p.lower() for p in img_path.parts]
            if any('forged' in p or 'fake' in p for p in parent_parts):
                is_forged = True
            elif any('authentic' in p or 'real' in p for p in parent_parts):
                is_authentic = True
        
        # Determine train/test split (you can adjust the ratio)
        # Simple approach: use existing folder structure or split by ratio
        is_test = 'test' in path_str or (authentic_count + forged_count) % 5 == 0
        
        # Determine destination
        if is_forged:
            if is_test:
                dest_dir = test_path / 'forged'
                test_count += 1
            else:
                dest_dir = train_path / 'forged'
                train_count += 1
            forged_count += 1
        elif is_authentic:
            if is_test:
                dest_dir = test_path / 'authentic'
                test_count += 1
            else:
                dest_dir = train_path / 'authentic'
                train_count += 1
            authentic_count += 1
        else:
            # If category can't be determined, put in a separate folder
            unknown_path = organized_path / 'unknown'
            unknown_path.mkdir(parents=True, exist_ok=True)
            dest_dir = unknown_path
            print(f"  Unknown category: {img_path.name}")
        
        # Copy file to destination
        dest_file = dest_dir / img_path.name
        
        # Handle duplicate filenames
        counter = 1
        while dest_file.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            dest_file = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(img_path, dest_file)
    
    print("\n" + "="*50)
    print("Dataset organization complete!")
    print("="*50)
    print(f"Total images processed: {len(all_images)}")
    print(f"Authentic images: {authentic_count}")
    print(f"Forged images: {forged_count}")
    print(f"Train set: {train_count}")
    print(f"Test set: {test_count}")
    print(f"\nOrganized dataset location: {organized_path}")
    print("="*50)
    
    return str(organized_path)

def print_directory_structure(path: str, max_depth: int = 3) -> None:
    """Print the directory structure"""
    path = Path(path)
    print(f"\nDirectory structure of {path}:")
    print("="*50)
    
    def print_tree(directory: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        contents = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, item in enumerate(contents):
            is_last = i == len(contents) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}", end="")
            
            if item.is_dir():
                # Count files in directory
                try:
                    file_count = len([f for f in item.rglob('*') if f.is_file()])
                    print(f" ({file_count} files)")
                except:
                    print()
                
                extension_prefix = "    " if is_last else "│   "
                print_tree(item, prefix + extension_prefix, depth + 1)
            else:
                print()
    
    print_tree(path)
    print("="*50)

def main():
    # Kaggle credentials
    KAGGLE_USERNAME = "anithavakkalagadda11"
    KAGGLE_KEY = "7d3a3308469a97260d966ad938dcb985"
    
    # Dataset information
    DATASET_NAME = "labid93/image-forgery-detection"
    DOWNLOAD_PATH = "./dataset"
    
    # Force download (set to True to re-download even if exists)
    FORCE_DOWNLOAD = False
    
    print("="*50)
    print("Image Forgery Detection Dataset Downloader")
    print("="*50)
    
    # Setup credentials
    setup_kaggle_credentials(KAGGLE_USERNAME, KAGGLE_KEY)
    
    # Download dataset (will skip if already exists)
    downloaded = download_dataset(DATASET_NAME, DOWNLOAD_PATH, FORCE_DOWNLOAD)
    
    # Check if dataset exists
    if not check_dataset_exists(DOWNLOAD_PATH):
        print("\n✗ Dataset not found after download attempt!")
        print("  Please check your Kaggle credentials and internet connection.")
        return
    
    # Show current directory structure
    print_directory_structure(DOWNLOAD_PATH, max_depth=2)
    
    # Organize dataset
    organized_path = organize_dataset(DOWNLOAD_PATH)
    
    if organized_path:
        # Show organized directory structure
        print_directory_structure(organized_path, max_depth=2)
        print(f"\n✓ Dataset ready at: {organized_path}")
        print("You can now proceed with training the models!")
    else:
        print("\n✗ Dataset organization failed!")
        print("  Please check the dataset structure manually.")

if __name__ == "__main__":
    main()
