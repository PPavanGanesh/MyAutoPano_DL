#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code
"""

import numpy as np
import cv2
import os
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def patch(image, x, y, patch_size):
    """Extract a patch from the image centered at (x, y)."""
    return image[x - patch_size // 2:x + patch_size // 2, y - patch_size // 2:y + patch_size // 2]

def data_generation(image, patch_size=128, perturbation_factor=32):
    """
    Generate original and warped patches with random perturbations.
    """
    height, width = image.shape[:2]
    roIheight, roIwidth = [150, height - 150], [150, width - 150]

    x = np.random.randint(roIheight[0], roIheight[1])
    y = np.random.randint(roIwidth[0], roIwidth[1])

    original_patch = patch(image, x, y, patch_size)

    corners = np.array([
        [x - patch_size // 2, y - patch_size // 2],
        [x - patch_size // 2, y + patch_size // 2],
        [x + patch_size // 2, y - patch_size // 2],
        [x + patch_size // 2, y + patch_size // 2]
    ], dtype=np.float32)

    perturbation = np.random.randint(-perturbation_factor, perturbation_factor, size=(4, 2)).astype(np.float32)
    perturbed_corners = corners + perturbation

    homography_matrix = cv2.getPerspectiveTransform(corners, perturbed_corners)
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    warped_patch = patch(warped_image, x, y, patch_size)

    original_patch = original_patch / 255.0
    warped_patch = warped_patch / 255.0

    return original_patch, warped_patch, perturbation, corners, homography_matrix

def dataset():
    """
    Generate a dataset of original and warped patches for both training and validation.
    """
    base_dir = '/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/'
    train_images_dir = os.path.join(base_dir, 'Train')
    val_images_dir = os.path.join(base_dir, 'Val')
    train_output_dir = os.path.join(base_dir, 'GeneratedData/Train')
    val_output_dir = os.path.join(base_dir, 'GeneratedData/Val')
    
    for sub_dir in ['Original', 'Warped', 'Labels']:
        os.makedirs(os.path.join(train_output_dir, sub_dir), exist_ok=True)
        os.makedirs(os.path.join(val_output_dir, sub_dir), exist_ok=True)
    
    train_perturbations, train_corners, train_homographies = [], [], []
    val_perturbations, val_corners, val_homographies = [], [], []

    num_train_images, patches_per_train_image = 5000, 8
    num_val_images, patches_per_val_image = 1000, 20

    with open(os.path.join(train_output_dir, 'Labels/train_names.txt'), 'w') as train_file:
        for i in range(1, num_train_images + 1):
            image_path = os.path.join(train_images_dir, f'{i}.jpg')
            print(f'Processing training image: {image_path}')
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Skipping image {image_path} (could not be loaded)")
                continue
            image = cv2.resize(image, (480, 480))
            
            for j in range(patches_per_train_image):
                original_patch, warped_patch, perturbation, corners, homography_matrix = data_generation(image)
                file_name = f'{i}_{j}.jpg'
                cv2.imwrite(os.path.join(train_output_dir, 'Original', file_name), original_patch * 255.0)
                cv2.imwrite(os.path.join(train_output_dir, 'Warped', file_name), warped_patch * 255.0)
                train_perturbations.append(perturbation)
                train_corners.append(corners)
                train_homographies.append(homography_matrix)
                train_file.write(file_name + '\n')
    
    np.save(os.path.join(train_output_dir, 'Labels/perturbations.npy'), np.array(train_perturbations))
    np.save(os.path.join(train_output_dir, 'Labels/corners.npy'), np.array(train_corners))
    np.save(os.path.join(train_output_dir, 'Labels/homography_matrices.npy'), np.array(train_homographies))

    with open(os.path.join(val_output_dir, 'Labels/val_names.txt'), 'w') as val_file:
        for i in range(1, num_val_images + 1):
            image_path = os.path.join(val_images_dir, f'{i}.jpg')
            print(f'Processing validation image: {image_path}')
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Skipping image {image_path} (could not be loaded)")
                continue
            image = cv2.resize(image, (480, 480))
            
            for j in range(patches_per_val_image):
                original_patch, warped_patch, perturbation, corners, homography_matrix = data_generation(image)
                file_name = f'{i}_{j}.jpg'
                cv2.imwrite(os.path.join(val_output_dir, 'Original', file_name), original_patch * 255.0)
                cv2.imwrite(os.path.join(val_output_dir, 'Warped', file_name), warped_patch * 255.0)
                val_perturbations.append(perturbation)
                val_corners.append(corners)
                val_homographies.append(homography_matrix)
                val_file.write(file_name + '\n')
    
    np.save(os.path.join(val_output_dir, 'Labels/perturbations.npy'), np.array(val_perturbations))
    np.save(os.path.join(val_output_dir, 'Labels/corners.npy'), np.array(val_corners))
    np.save(os.path.join(val_output_dir, 'Labels/homography_matrices.npy'), np.array(val_homographies))
    
    print('Data generation completed for both training and validation.')

if __name__ == "__main__":
    dataset()
