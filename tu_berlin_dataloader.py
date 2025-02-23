import torch
import os
import argparse
import numpy as np

from collections import defaultdict

import torchvision.transforms as transforms
from PIL import Image
from glob import glob

from utils.tu_berlin_statistics import MEAN, STD
from utils.class_to_labels import classes_to_indices

from sklearn.model_selection import train_test_split, StratifiedKFold



class SketchData(torch.utils.data.Dataset):
    def __init__(self, image_indices, path_to_data, mode='train', invert_pixels=False):
        super(SketchData, self).__init__()

        self.mode = mode

        self.img_paths = glob(os.path.join(os.getcwd(), path_to_data, "**", "*.png"), recursive=True)

        self.indices = image_indices
        
        # Map indices to valid range
        self.index_map = {i: idx for i, idx in enumerate(self.indices)}

        # Define image transformations
        self.img_transform = self.get_image_transform(invert_pixels)

        # Get class labels for each image
        self.labels, self.labels_mapping = classes_to_indices(os.path.join(os.getcwd(), path_to_data))

        # Create a dictionary mapping each class to its image paths
        class_to_paths = defaultdict(list)
        for path in self.img_paths:
            class_name = os.path.basename(os.path.dirname(path))
            class_to_paths[class_name].append(path)

        # Ensure that each class has exactly 80 images
        for class_name, paths in class_to_paths.items():
            assert len(paths) == 80, f"Class {class_name} does not have exactly 80 images."
    
    def get_class_from_path(self, path):
        class_name = os.path.basename(os.path.dirname(path))
        return self.labels_mapping[class_name]
        
    def get_image_transform(self, invert_pixels):
        if self.mode != 'train':
            return transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

        base_transforms = [
            transforms.Resize([512, 512]),
            transforms.ToTensor()
        ]

        # 1. Tremor-like variations (milder than medical dataset)
        tremor_variations = transforms.RandomChoice([
            # Very mild
            transforms.ElasticTransform(alpha=20.0, sigma=3.0, fill=255),
            # Mild
            transforms.ElasticTransform(alpha=40.0, sigma=3.0, fill=255),
            # Moderate
            transforms.ElasticTransform(alpha=60.0, sigma=3.0, fill=255)
        ])

        # 2. Line quality variations
        line_quality = transforms.RandomChoice([
            # Sharp lines
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1)),
            # Normal lines
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1, 2)),
            # Blurry lines
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(2, 3))
        ])

        # 3. Discontinuity simulation
        discontinuity = transforms.RandomChoice([
            # Tiny gaps (like natural drawing)
            transforms.RandomErasing(p=0.3, scale=(0.01, 0.02), ratio=(0.1, 10.0)),
            # Small gaps (like tremor)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.03), ratio=(0.1, 10.0))
        ])

        # 4. General sketch variations (important for TU-Berlin diversity)
        sketch_variations = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])

        augmentation_pipeline = transforms.Compose([
            *base_transforms,
            transforms.RandomApply([tremor_variations], p=0.3),  # Lower probability for pre-training
            transforms.RandomApply([line_quality], p=0.4),  # Moderate probability
            transforms.RandomApply([discontinuity], p=0.2),  # Low probability
            transforms.RandomApply([sketch_variations], p=0.5),  # Higher probability for general variations
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        if invert_pixels:
            augmentation_pipeline.transforms.insert(-1, transforms.RandomSolarize(threshold=0, p=1))

        return augmentation_pipeline


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
            
        # Get the actual index from our mapping
        actual_idx = self.index_map[idx]
        
        # Get the image path
        img_path = self.img_paths[actual_idx]
        
        # Load and transform image
        img = Image.open(img_path)
        img = self.img_transform(img)
        
        # Get class label
        label = self.get_class_from_path(img_path)
        
        return img, label


def get_test_and_folds(root_dir, seed):
    img_paths = glob(os.path.join(os.getcwd(), root_dir, "**", "*.png"), recursive=True)
    #img_paths = sorted(img_paths)

    # Get class labels for stratification
    _, class_name_to_label = classes_to_indices(os.path.join(os.getcwd(), root_dir))
    labels = []
    # class_to_paths = defaultdict(list)
    for path in img_paths:
        class_name = os.path.basename(os.path.dirname(path))
        labels.append(class_name_to_label[class_name])

    # print("done")
    # First split data into train+val and test sets (80-20 split)
    train_val_indices, test_indices, _, _ = train_test_split(
        np.arange(len(img_paths)),
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed
    )

    # Create test dataset
    test_dataset = SketchData(image_indices=test_indices, path_to_data=root_dir, mode='test', invert_pixels=False)

    # Create stratified k-fold for remaining data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Get labels for remaining data for stratification
    train_val_labels = [labels[i] for i in train_val_indices]

    # Create folds
    folds = list(skf.split(train_val_indices, train_val_labels))

    return test_dataset, folds, train_val_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', nargs="?", type=str, default="TU-berlin-dataset",
                        help="repository of tuBerlin dataset")
    parser.add_argument('--get_test_set', nargs="?", type=str, default="Test",
                        help="get test or train or val indices")
    parser.add_argument('--seed', nargs="?", type=int, default=42,
                        help="get test or train or val indices")                    
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(os.getcwd(), args.data_dir)):
        print(f"no data repository found")

    path_to_data = args.data_dir

    test_set, folds, train_val_indices = get_test_and_folds(root_dir=path_to_data, seed=args.seed)
    
    test_indices = test_set.indices
    
    if args.get_test_set == 'Test':
        for idx in test_indices:
            # actual_idx = test_set.index_map[idx]
            # Get the image path
            img_path = test_set.img_paths[idx]
            print(("/").join(img_path.split("/")[4:]))
        exit()
        
    # for fold_idx, (train_idx, val_idx) in enumerate(folds):
        
        # if args.fold_idx != fold_idx:
            # continue
        
        # Map fold indices back to original dataset indices
        # train_indices = train_val_indices[train_idx]
        # val_indices = train_val_indices[val_idx]

        # train_set = SketchData2(image_indices=train_indices, path_to_data=path_to_data, mode='train',invert_pixels=False)
        # val_set = SketchData2(image_indices=val_indices, path_to_data=path_to_data, mode='val', invert_pixels=False)

        # print(f"Fold {fold_idx + 1}:")
        # print(f"Training set size: {len(train_set)}")
        # print(f"Validation set size: {len(val_set)}")
        # print(f"Test set size: {len(test_set)}")
        # train_indices = train_set.indices
        
        # if args.get_test_set == 'Train':
            # for idx in train_indices:
                # print(idx)
            # exit()
        
        # val_indices = val_set.indices
        
        # if args.get_test_set == 'Val':
            # for idx in val_indices:
                # print(idx)
            # exit()

        
