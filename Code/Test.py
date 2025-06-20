import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from Network_ import HomographyModel

def ReadPatches(original_path, warped_path):
    original_patch = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    warped_patch = cv2.imread(warped_path, cv2.IMREAD_GRAYSCALE)

    if original_patch is None or warped_patch is None:
        print(f"ERROR: Could not read patches: {original_path}, {warped_path}")
        sys.exit()

    original_patch = original_patch / 255.0
    warped_patch = warped_patch / 255.0

    original_patch = np.expand_dims(original_patch, axis=0)
    warped_patch = np.expand_dims(warped_patch, axis=0)

    original_patch = torch.tensor(original_patch, dtype=torch.float32).unsqueeze(0)
    warped_patch = torch.tensor(warped_patch, dtype=torch.float32).unsqueeze(0)

    return original_patch, warped_patch

def draw_corners(image, corners, color, thickness=2):
    """Draw quadrilateral with correct corner ordering"""
    corners = corners.astype(int).reshape(-1, 2)
    # Reorder corners: [top-left, top-right, bottom-right, bottom-left]
    ordered_corners = corners[[0, 1, 3, 2], :]
    
    # Draw quadrilateral
    cv2.polylines(image, [ordered_corners], isClosed=True, color=color, thickness=thickness)
    # Draw corner points
    for pt in ordered_corners:
        cv2.circle(image, tuple(pt), radius=4, color=color, thickness=-1)
    return image

def visualize_results(original_image, original_corners, perturbed_corners, predicted_corners, output_path):
    """Visualize all three corner sets on the original color image"""
    # Convert to BGR if needed (preserve color channels)
    if len(original_image.shape) == 2:
        display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = original_image.copy()
    
    # Draw original corners (blue)
    display_image = draw_corners(display_image, original_corners, (255, 0, 0))
    
    # Draw perturbed corners (green)
    display_image = draw_corners(display_image, perturbed_corners, (0, 255, 0))
    
    # Draw predicted corners (red)
    display_image = draw_corners(display_image, predicted_corners, (0, 0, 255))
    
    # Add legend
    cv2.putText(display_image, "Original (Blue)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    cv2.putText(display_image, "Perturbed (Green)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.putText(display_image, "Predicted (Red)", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    # cv2.putText(display_image, "Original (Blue)", (10, 35), 
    #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255, 0, 0), 2)
    # cv2.putText(display_image, "Perturbed (Green)", (10, 70), 
    #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 2)
    # cv2.putText(display_image, "Predicted (Red)", (10, 105), 
    #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, display_image)

def TestOperation(ModelPath, OriginalDir, WarpedDir, PerturbationsPath, 
                 NamesFile, OutputFile, ValImagesDir, ValCornersPath):
    # Load model
    model = HomographyModel(model_type="Sup").to('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ModelPath)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load ground truth data
    ground_truth_perturbations = np.load(PerturbationsPath)
    val_corners = np.load(ValCornersPath)
    
    with open(NamesFile, 'r') as f:
        test_names = [line.strip() for line in f]

    # Create visualization directory
    vis_dir = os.path.join("20Visualizations_SUP_Test_Data")
    os.makedirs(vis_dir, exist_ok=True)

    with open(OutputFile, "w") as output_file:
        total_l2_error = 0.0

        for i, file_name in tqdm(enumerate(test_names), total=len(test_names)):
            # Process patches
            original_path = os.path.join(OriginalDir, file_name)
            warped_path = os.path.join(WarpedDir, file_name)
            original_patch, warped_patch = ReadPatches(original_path, warped_path)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # print("Deviceeeeeeeeee", device)
            original_patch = original_patch.to(device)
            warped_patch = warped_patch.to(device)

            # Get ground truth data
            gt_perturbation = ground_truth_perturbations[i]
            original_corners = val_corners[i]
            
            # Predict perturbation
            input_batch = torch.cat((original_patch, warped_patch), dim=1)
            with torch.no_grad():
                predicted_perturbation = model(input_batch).squeeze().cpu().numpy()

            # Compute L2 error
            gt_tensor = torch.tensor(gt_perturbation.reshape(-1), dtype=torch.float32).to(device)
            pred_tensor = torch.tensor(predicted_perturbation, dtype=torch.float32).to(device)
            l2_error = torch.norm(pred_tensor - gt_tensor).item()
            total_l2_error += l2_error

            # Visualization
            img_idx = int(file_name.split('_')[0])
            original_image = cv2.imread(os.path.join(ValImagesDir, f"{img_idx}.jpg"))
            
            if original_image is None:
                print(f"Could not load original image: {img_idx}.jpg")
                continue

            # Calculate all corner sets
            perturbed_corners = original_corners + gt_perturbation.reshape(4, 2)
            predicted_corners = original_corners + predicted_perturbation.reshape(4, 2)
            
            # Save visualization
            output_path = os.path.join(vis_dir, f"{file_name.split('.')[0]}_vis.jpg")
            visualize_results(
                original_image=original_image,
                original_corners=original_corners,
                perturbed_corners=perturbed_corners,
                predicted_corners=predicted_corners,
                output_path=output_path
            )

            # Write prediction
            output_file.write(",".join(map(str, predicted_perturbation)) + "\n")

        avg_l2_error = total_l2_error / len(test_names)
        print(f"Average L2 Error on Test Set: {avg_l2_error:.4f}")


def main():
    
    # Update these paths according to your setup
    
    OriginalDir = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/GeneratedData/Test/Original"
    WarpedDir = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/GeneratedData/Test/Warped"
    PerturbationsPath = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/GeneratedData/Test/Labels/perturbations.npy"
    NamesFile = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/GeneratedData/Test/Labels/test_names.txt"
    ValImagesDir = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/Test"
    ValCornersPath = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/GeneratedData/Test/Labels/corners.npy"
    
    ModelPath = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Code/Logs/experiment_20250208_132947_SUP40/experiment_20250208_143952/checkpoint_epoch_20.ckpt"
    OutputFile = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/GeneratedData/Test/SUP_Predictions_Test.txt"
    

    TestOperation(
        ModelPath=ModelPath,
        OriginalDir=OriginalDir,
        WarpedDir=WarpedDir,
        PerturbationsPath=PerturbationsPath,
        NamesFile=NamesFile,
        OutputFile=OutputFile,
        ValImagesDir=ValImagesDir,
        ValCornersPath=ValCornersPath
    )

if __name__ == "__main__":
    main()
