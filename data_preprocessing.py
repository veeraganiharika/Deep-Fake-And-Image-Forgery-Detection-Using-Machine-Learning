import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

class DataPreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        return img
    
    def extract_ela_features(self, image_path, quality=90):
        """Extract Error Level Analysis features"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Save with compression
            temp_path = 'temp_ela.jpg'
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Load compressed image
            compressed = cv2.imread(temp_path)
            
            # Calculate difference
            ela = cv2.absdiff(img, compressed)
            ela = cv2.resize(ela, self.img_size)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return ela
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset_from_folder(self, folder_path, use_ela=True):
        """
        Load images from a single folder (either train or test)
        Expected structure: folder_path/authentic/*.jpg and folder_path/forgery/*.jpg
        """
        folder_path = Path(folder_path)
        
        images = []
        labels = []
        
        # Define class mappings - check for multiple possible folder names
        class_folders = {
            'authentic': ['authentic', 'Authentic', 'Au', 'real', 'Real'],
            'forgery': ['forgery', 'Forgery', 'forged', 'Forged', 'Tp', 'fake', 'Fake']
        }
        
        print(f"Loading dataset from: {folder_path}")
        
        # Load authentic images (label 0)
        authentic_loaded = False
        for folder_name in class_folders['authentic']:
            class_dir = folder_path / folder_name
            if class_dir.exists() and class_dir.is_dir():
                print(f"Found authentic images in: {class_dir}")
                count = self._load_images_from_class(class_dir, 0, images, labels, use_ela)
                print(f"  Loaded {count} authentic images")
                authentic_loaded = True
                break
        
        if not authentic_loaded:
            print(f"Warning: No authentic folder found in {folder_path}")
        
        # Load forgery images (label 1)
        forgery_loaded = False
        for folder_name in class_folders['forgery']:
            class_dir = folder_path / folder_name
            if class_dir.exists() and class_dir.is_dir():
                print(f"Found forgery images in: {class_dir}")
                count = self._load_images_from_class(class_dir, 1, images, labels, use_ela)
                print(f"  Loaded {count} forgery images")
                forgery_loaded = True
                break
        
        if not forgery_loaded:
            print(f"Warning: No forgery folder found in {folder_path}")
        
        return images, labels
    
    def _load_images_from_class(self, class_dir, label, images, labels, use_ela):
        """Helper function to load images from a specific class directory"""
        count = 0
        
        # Support multiple image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(ext)))
        
        for img_path in image_files:
            if use_ela:
                img = self.extract_ela_features(img_path)
            else:
                img = self.load_image(img_path)
            
            if img is not None:
                images.append(img)
                labels.append(label)
                count += 1
        
        return count
    
    def load_dataset(self, dataset_path, use_ela=True):
        """
        Load complete dataset with train/test structure
        Expected structure:
        dataset_path/
            train/
                authentic/
                forgery/
            test/
                authentic/
                forgery/
        """
        dataset_path = Path(dataset_path)
        
        print("="*60)
        print("Loading Dataset")
        print("="*60)
        
        # Check if train and test folders exist
        train_path = dataset_path / 'train'
        test_path = dataset_path / 'test'
        
        if not train_path.exists():
            raise ValueError(f"Training folder not found: {train_path}")
        if not test_path.exists():
            raise ValueError(f"Test folder not found: {test_path}")
        
        # Load training data
        print("\n[TRAIN SET]")
        X_train_list, y_train_list = self.load_dataset_from_folder(train_path, use_ela)
        
        # Load test data
        print("\n[TEST SET]")
        X_test_list, y_test_list = self.load_dataset_from_folder(test_path, use_ela)
        
        # Convert to numpy arrays and normalize
        if len(X_train_list) == 0:
            raise ValueError("No training images loaded!")
        if len(X_test_list) == 0:
            raise ValueError("No test images loaded!")
        
        X_train = np.array(X_train_list, dtype=np.float32) / 255.0
        y_train = np.array(y_train_list)
        X_test = np.array(X_test_list, dtype=np.float32) / 255.0
        y_test = np.array(y_test_list)
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Training set: {len(X_train)} images")
        print(f"  - Authentic: {np.sum(y_train == 0)}")
        print(f"  - Forgery: {np.sum(y_train == 1)}")
        print(f"\nTest set: {len(X_test)} images")
        print(f"  - Authentic: {np.sum(y_test == 0)}")
        print(f"  - Forgery: {np.sum(y_test == 1)}")
        print(f"\nImage shape: {X_train.shape[1:]}")
        print(f"Using ELA features: {use_ela}")
        print("="*60)
        
        return X_train, X_test, y_train, y_test
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        Note: This method is kept for backward compatibility,
        but with your folder structure, you don't need to use this
        """
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, save_dir):
        """Save preprocessed data as numpy arrays"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        np.save(save_dir / 'X_train.npy', X_train)
        np.save(save_dir / 'X_test.npy', X_test)
        np.save(save_dir / 'y_train.npy', y_train)
        np.save(save_dir / 'y_test.npy', y_test)
        
        print(f"\nPreprocessed data saved to: {save_dir}")
        print(f"  - X_train.npy: {X_train.shape}")
        print(f"  - X_test.npy: {X_test.shape}")
        print(f"  - y_train.npy: {y_train.shape}")
        print(f"  - y_test.npy: {y_test.shape}")
    
    def load_preprocessed_data(self, save_dir):
        """Load preprocessed data from numpy arrays"""
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            raise ValueError(f"Directory not found: {save_dir}")
        
        X_train = np.load(save_dir / 'X_train.npy')
        X_test = np.load(save_dir / 'X_test.npy')
        y_train = np.load(save_dir / 'y_train.npy')
        y_test = np.load(save_dir / 'y_test.npy')
        
        print(f"Loaded preprocessed data from: {save_dir}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor(img_size=(224, 224))
    
    # Set your dataset path
    dataset_path = r"E:\desktop\Dite\forgey\dataset"
    
    # Option 1: Load data with ELA features
    print("\nLoading dataset with ELA features...")
    X_train, X_test, y_train, y_test = preprocessor.load_dataset(
        dataset_path, 
        use_ela=True
    )
    
    # Save preprocessed data
    save_dir = r"E:\desktop\Dite\forgey\preprocessed_data"
    preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test, save_dir)
    
    # Option 2: Load without ELA (just resized RGB images)
    # X_train, X_test, y_train, y_test = preprocessor.load_dataset(
    #     dataset_path, 
    #     use_ela=False
    # )
    
    # Later, you can load the preprocessed data directly
    # X_train, X_test, y_train, y_test = preprocessor.load_preprocessed_data(save_dir)
