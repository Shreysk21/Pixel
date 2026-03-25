"""
Data Augmentation Module
Implements various augmentation techniques for object detection datasets
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentor:
    """Apply augmentation transformations to dataset"""
    
    def __init__(self, img_size=(448, 448)):
        """
        Initialize augmentor
        Args:
            img_size: Target image size (height, width)
        """
        self.img_size = img_size
        self.augmentation_count = 0
    
    def get_train_augmentations(self):
        """
        Get augmentation pipeline for training
        Returns:
            Composed albumentations augmentation pipeline
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            A.Resize(self.img_size[0], self.img_size[1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def get_val_augmentations(self):
        """
        Get augmentation pipeline for validation (minimal augmentation)
        Returns:
            Composed albumentations augmentation pipeline
        """
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            A.Resize(self.img_size[0], self.img_size[1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def augment_image(self, image, bboxes=None, class_labels=None):
        """
        Apply augmentation to single image
        Args:
            image: Input image (numpy array or path)
            bboxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
            class_labels: List of class labels for bboxes
        Returns:
            Augmented image and updated bboxes
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform = self.get_train_augmentations()
        
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []
        
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        self.augmentation_count += 1
        logger.info(f"Augmented image {self.augmentation_count}")
        
        return augmented['image'], augmented.get('bboxes', []), augmented.get('class_labels', [])
    
    def augment_dataset(self, image_dir, output_dir, augmentations_per_image=3):
        """
        Apply augmentation to entire dataset
        Args:
            image_dir: Directory containing original images
            output_dir: Directory to save augmented images
            augmentations_per_image: Number of augmented versions per image
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_file in image_dir.rglob('*'):
            if image_file.suffix.lower() not in image_extensions:
                continue
            
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"Failed to read: {image_file}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Save original
            output_path = output_dir / f"{image_file.stem}_original{image_file.suffix}"
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Create augmented versions
            transform = self.get_train_augmentations()
            
            for aug_idx in range(augmentations_per_image):
                try:
                    augmented = transform(image=image)
                    aug_image = augmented['image']
                    
                    # Save augmented image
                    output_path = output_dir / f"{image_file.stem}_aug_{aug_idx}{image_file.suffix}"
                    
                    # Convert tensor back to numpy if needed
                    if isinstance(aug_image, np.ndarray):
                        aug_image = (aug_image * 255).astype(np.uint8)
                        cv2.imwrite(str(output_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    logger.info(f"Created augmentation {aug_idx+1}/{augmentations_per_image} for {image_file.name}")
                
                except Exception as e:
                    logger.error(f"Failed to augment {image_file}: {str(e)}")
        
        logger.info(f"Dataset augmentation complete. Saved to {output_dir}")
    
    def random_crop_augmentation(self, image, crop_size=(384, 384), p=0.5):
        """
        Apply random crop augmentation
        Args:
            image: Input image
            crop_size: Size of crop (height, width)
            p: Probability of applying augmentation
        Returns:
            Cropped image
        """
        if np.random.rand() > p:
            return image
        
        transform = A.Compose([
            A.RandomCrop(height=crop_size[0], width=crop_size[1]),
        ])
        
        return transform(image=image)['image']
    
    def mosaic_augmentation(self, images, output_size=(416, 416)):
        """
        Apply mosaic augmentation (combine 4 images)
        Args:
            images: List of 4 images
            output_size: Output size (height, width)
        Returns:
            Mosaic augmented image
        """
        if len(images) != 4:
            raise ValueError("Mosaic augmentation requires exactly 4 images")
        
        # Resize each image to half of output size
        h, w = output_size
        half_h, half_w = h // 2, w // 2
        
        mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        
        positions = [(0, 0), (0, half_w), (half_h, 0), (half_h, half_w)]
        
        for img, (y, x) in zip(images, positions):
            resized = cv2.resize(img, (half_w, half_h))
            mosaic[y:y+half_h, x:x+half_w] = resized
        
        logger.info("Mosaic augmentation applied")
        return mosaic
    
    def get_augmentation_stats(self):
        """Get augmentation statistics"""
        return {
            'total_augmentations_applied': self.augmentation_count
        }


class AugmentationPipeline:
    """Complete augmentation pipeline for dataset"""
    
    def __init__(self, img_size=(448, 448)):
        self.augmentor = DataAugmentor(img_size=img_size)
    
    def create_augmented_dataset(self, source_dir, output_dir, 
                                augmentations_per_image=3, split_ratio=0.8):
        """
        Create augmented dataset with train/val split
        Args:
            source_dir: Source dataset directory
            output_dir: Output directory for augmented dataset
            augmentations_per_image: Number of augmentations per image
            split_ratio: Train/val split ratio
        """
        logger.info("Creating augmented dataset...")
        
        output_dir = Path(output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        self.augmentor.augment_dataset(source_dir, train_dir, augmentations_per_image)
        
        logger.info(f"Augmented dataset created at {output_dir}")


def main():
    """Example usage of DataAugmentor"""
    augmentor = DataAugmentor(img_size=(448, 448))
    
    # Example: Augment dataset
    # augmentor.augment_dataset(
    #     image_dir="path/to/images",
    #     output_dir="path/to/augmented_images",
    #     augmentations_per_image=3
    # )
    
    logger.info("DataAugmentor initialized and ready to use")


if __name__ == "__main__":
    main()
