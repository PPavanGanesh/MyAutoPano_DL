import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from Network_ import HomographyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """
    Load the trained homography model.
    """
    model = HomographyModel(model_type="Sup").to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    return model

def predict_homography(model, img1_gray, img2_gray):
    """
    Predict the homography between img1 and img2 using the trained model.
    """
    # Resize to 128x128 and normalize
    img1_resized = cv2.resize(img1_gray, (128, 128)) / 255.0
    img2_resized = cv2.resize(img2_gray, (128, 128)) / 255.0

    # Prepare input for the model
    img1_resized = np.expand_dims(img1_resized, axis=0)
    img2_resized = np.expand_dims(img2_resized, axis=0)
    input_batch = np.concatenate((img1_resized, img2_resized), axis=0).astype(np.float32)  # (2, 128, 128)
    input_batch = torch.tensor(input_batch).unsqueeze(0).permute(0, 1, 2, 3).to(device)  # (1, 2, 128, 128)

    # Predict displacements
    predicted_displacements = model(input_batch).detach().cpu().numpy().reshape(4, 2)

    # Compute the homography matrix
    original_corners = np.array([[0, 0], [0, 128], [128, 0], [128, 128]], dtype=np.float32)
    predicted_corners = original_corners + predicted_displacements
    H, _ = cv2.findHomography(original_corners, predicted_corners)
    return H

def get_warped_image(img1, img2, H):
    """
    Warp img2 to align with img1 using the predicted homography H.
    """
    height, width = img1.shape[:2]
    warped_img2 = cv2.warpPerspective(img2, H, (width * 2, height))
    return warped_img2

def blend_images(img1, img2):
    """
    Blend img1 and img2 using alpha blending.
    """
    alpha = 0.5
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    return blended

def stitch_images(images, model):
    """
    Stitch a sequence of images into a panorama using the trained model.
    """
    # Initialize the panorama with the first image
    stitched_panorama = images[0]

    for i in range(len(images) - 1):
        print(f"Stitching image {i+1} and image {i+2}...")

        # Convert the current panorama and next image to grayscale
        gray_pano = cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        # Predict the homography between the current panorama and the next image
        H = predict_homography(model, gray_pano, gray_next)

        # Warp the next image using the predicted homography
        warped_img = get_warped_image(stitched_panorama, images[i + 1], H)

        # Blend the current panorama with the warped image
        stitched_panorama = blend_images(stitched_panorama, warped_img)

        # Display intermediate results
        plt.imshow(cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2RGB))
        plt.title(f"Stitched Image {i + 1}")
        plt.axis('off')
        plt.show()

    return stitched_panorama

def main():
    # Load the trained model
    checkpoint_path = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Code/Logs/experiment_20250208_132947_SUP40/experiment_20250208_143952/checkpoint_epoch_39.ckpt"
    model = load_model(checkpoint_path)

    # Path to images
    image_path = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data/Phase2Pano"
    images = [cv2.imread(os.path.join(image_path, f"{i}.jpg")) for i in range(len(os.listdir(image_path)))]

    # Resize images for faster processing (optional)
    # images = [cv2.resize(img, (800, 600)) for img in images]

    # Stitch the images
    panorama = stitch_images(images, model)

    # Display and save the final panorama
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Final Panorama")
    plt.axis('off')
    plt.show()

    # Save the result
    output_path = r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Code/results/panorama.jpg"
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved at: {output_path}")

if __name__ == "__main__":
    main()
