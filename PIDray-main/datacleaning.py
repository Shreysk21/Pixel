"""
Data Cleaning Module
Handles removal of corrupted images, invalid annotations, and data validation
"""

import os
import json
import cv2
from PIL import Image
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean dataset by removing corrupted files and fixing annotations"""
    
    def __init__(self, data_dir):
        """
        Initialize the data cleaner
        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = Path(data_dir)
        self.removed_files = []
        self.invalid_annotations = []
    
    def check_image_integrity(self, image_path):
        """
        Verify if image file is valid and readable
        Args:
            image_path: Path to image file
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception as e:
            logger.warning(f"Invalid image: {image_path} - {str(e)}")
            return False
    
    def check_annotation_validity(self, annotation_path):
        """
        Verify if annotation file is valid and properly formatted
        Args:
            annotation_path: Path to annotation file
        Returns:
            bool: True if annotation is valid, False otherwise
        """
        try:
            if annotation_path.suffix == '.json':
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                    if 'annotations' in data or 'images' in data:
                        return True
            return False
        except Exception as e:
            logger.warning(f"Invalid annotation: {annotation_path} - {str(e)}")
            return False
    
    def remove_corrupted_images(self, image_dir):
        """
        Remove corrupted image files from directory
        Args:
            image_dir: Directory containing images
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_path = Path(image_dir)
        
        for img_file in image_path.rglob('*'):
            if img_file.suffix.lower() in image_extensions:
                if not self.check_image_integrity(img_file):
                    try:
                        os.remove(img_file)
                        self.removed_files.append(str(img_file))
                        logger.info(f"Removed corrupted image: {img_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove {img_file}: {str(e)}")
    
    def validate_annotations(self, annotation_dir):
        """
        Validate and fix annotation files
        Args:
            annotation_dir: Directory containing annotation files
        """
        annotation_path = Path(annotation_dir)
        
        for ann_file in annotation_path.rglob('*.json'):
            if not self.check_annotation_validity(ann_file):
                self.invalid_annotations.append(str(ann_file))
                logger.warning(f"Invalid annotation: {ann_file}")
    
    def remove_unlabeled_images(self, image_dir, annotation_dir):
        """
        Remove images that don't have corresponding annotations
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
        """
        image_path = Path(image_dir)
        annotation_path = Path(annotation_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        annotated_files = set()
        for ann_file in annotation_path.rglob('*.json'):
            annotated_files.add(ann_file.stem)
        
        for img_file in image_path.rglob('*'):
            if img_file.suffix.lower() in image_extensions:
                if img_file.stem not in annotated_files:
                    try:
                        os.remove(img_file)
                        self.removed_files.append(str(img_file))
                        logger.info(f"Removed unlabeled image: {img_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove {img_file}: {str(e)}")
    
    def clean_dataset(self, image_dir, annotation_dir=None):
        """
        Perform complete dataset cleaning
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations (optional)
        """
        logger.info("Starting dataset cleaning...")
        
        # Check image integrity
        logger.info("Checking image integrity...")
        self.remove_corrupted_images(image_dir)
        
        # Validate annotations if provided
        if annotation_dir:
            logger.info("Validating annotations...")
            self.validate_annotations(annotation_dir)
            
            # Remove unlabeled images
            logger.info("Removing unlabeled images...")
            self.remove_unlabeled_images(image_dir, annotation_dir)
        
        logger.info(f"Cleaning complete. Removed {len(self.removed_files)} files.")
        logger.info(f"Found {len(self.invalid_annotations)} invalid annotations.")
        
        return {
            'removed_files': self.removed_files,
            'invalid_annotations': self.invalid_annotations
        }
    
    def generate_report(self, output_file='cleaning_report.json'):
        """Generate cleaning report"""
        report = {
            'removed_files': self.removed_files,
            'invalid_annotations': self.invalid_annotations,
            'total_removed': len(self.removed_files),
            'total_invalid_annotations': len(self.invalid_annotations)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Report saved to {output_file}")
        return report


def main():
    """Example usage of DataCleaner"""
    # Example paths - modify according to your dataset structure
    image_directory = "path/to/images"
    annotation_directory = "path/to/annotations"
    
    cleaner = DataCleaner(data_dir=".")
    report = cleaner.clean_dataset(image_directory, annotation_directory)
    cleaner.generate_report()


if __name__ == "__main__":
    main()
