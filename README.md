# MyAutoPano - Phase 2 📸🔁🧠

A deep learning-based pipeline for estimating homographies between image patches and stitching multiple images into a seamless panorama using supervised and unsupervised learning approaches.

---

## 🚀 Project Overview

This project extends traditional image stitching by replacing handcrafted methods with **CNN-based homography estimation**. Using a dataset of perturbed image patches, we train supervised and unsupervised models to predict homography transformations, ultimately allowing for robust panorama creation.

Key features include:
- 🔧 Synthetic data generation with known ground truth
- 🧠 Supervised and unsupervised CNN models
- 📊 Visualization and evaluation on test images
- 🖼️ Stitching multiple images into panoramic views

---

## 📁 Directory Structure

```

├── Data\_Generation.py          # Generate training & validation patch datasets
├── Train.py                    # Train supervised or unsupervised models
├── Test.py                     # Evaluate trained model on test data
├── Wrapper.py                  # Stitch images using trained homography model
├── Network.py                  # CNN architectures for both Sup/UnSup modes
├── Logs/                       # Checkpoints and loss graphs
├── Data/                       # Folder for raw and generated data
│   ├── Train/
│   ├── Val/
│   └── GeneratedData/

````

---

## 🧠 Deep Learning Methods

### 1. **Supervised Learning**  
Predicts the 8-dimensional displacement vector between original and perturbed patch corners.

- Loss Function: MSE
- Model: 8-layer CNN + 2 fully connected layers
- Output: 4 corner displacements → Homography matrix

### 2. **Unsupervised Learning**  
Estimates homography indirectly using photometric error between the warped image and the target.

- Loss Function: L1 photometric loss
- Module: TensorDLT (to compute 3×3 homography matrix)
- Library: Kornia for differentiable warping

---

## 🧪 Phase Breakdown

### 🔹 **Data Generation** – `Data_Generation.py`

- Extract 128×128 patches from training/validation images
- Apply random corner perturbations (±32 pixels)
- Save original, warped, perturbation vectors and corner locations

### 🔹 **Training** – `Train.py`

- Supports both Supervised (`--ModelType Sup`) and Unsupervised (`--ModelType Unsup`) modes
- TensorBoard logging and checkpoint saving
- Example:

```bash
python3 Train.py --BasePath /path/to/Data --NumEpochs 40 --MiniBatchSize 32 --ModelType Sup
````

### 🔹 **Testing** – `Test.py`

* Loads trained model, compares predicted and ground-truth displacements
* Visualizes corner sets (original, perturbed, predicted) for verification
* Outputs L2 error per sample

### 🔹 **Panorama Stitching** – `Wrapper.py`

* Uses predicted homography to warp and align consecutive images
* Displays and saves final stitched panorama

---

## 📊 Sample Results

* 🧠 Train Set EPE: **7.62**
* 📊 Validation EPE: **7.62**
* 🧪 Test Set EPE: **7.66**
* 📸 Panorama blending with multiple images using learned transformations

---

## ⚙️ Requirements

* Python 3.7+
* PyTorch
* OpenCV
* Kornia
* Matplotlib
* NumPy
* Pandas
* TensorBoard (for training visualization)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* All patches are grayscale, normalized to \[0,1]
* Models are trained on 40,000 samples and validated on 20,000 samples
* Use absolute paths for checkpoints and data directories

---

## 👨‍💻 Authors

* Pavan Ganesh Pabbineedi – [WPI Robotics](ppabbineedi@wpi.edu)
* Manideep Duggi – [WPI Robotics](mduggi@wpi.edu)

---

## 📄 License

This project is for academic use only. Please contact the authors for other use cases.

```

