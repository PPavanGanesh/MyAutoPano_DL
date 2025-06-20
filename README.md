Project Title: MyAutoPano - Phase 2
Classical and Deep Learning Approaches for Geometric Computer Vision
This project focuses on training a homography estimation model using supervised and unsupervised techniques for geometric computer vision applications. The provided scripts help generate datasets of original and warped image patches and train the model on these datasets.

Step 1: Generating the Dataset
The dataset generation process extracts patches from images, applies random perturbations, and stores the original and warped patches along with ground-truth perturbation vectors.
Command to Generate Data
python Data_generation.py

Base Directory: Ensure that Data/Train and Data/Val contain the original grayscale images.
Generated Data: The generated patches and labels will be saved under GeneratedData/Train and GeneratedData/Val.
Output Structure:
|-- GeneratedData/
    |-- Train/
        |-- Original/       # Original patches
        |-- Warped/         # Warped patches
        |-- Labels/         # Perturbations, corners, and homography matrices
    |-- Val/
        |-- Original/
        |-- Warped/
        |-- Labels/


Step 2: Training the Model
The training script supports both supervised and unsupervised learning approaches for homography estimation. It uses TensorBoard for visualizing the training progress and saves model checkpoints periodically.
Command to Train the Model
python train.py --BasePath <path_to_data> --LogsPath <logs_directory> --NumEpochs 10 --MiniBatchSize 32 --ModelType Sup

Command-Line Arguments:
Argument
Description
Default
--BasePath
Path to the dataset (original and generated data)
Eg. C:\Users\pavan\OneDrive\Desktop\CV\YourDirectoryID_p1\Phase2\Data
--LogsPath
Path to save TensorBoard logs and checkpoints
Logs/
--NumEpochs
Number of epochs to train the model
10
--MiniBatchSize
Batch size used during training
32
--ModelType
Type of model: Sup for supervised or Unsup for unsupervised
Sup
--LoadCheckPoint
Resume from an existing checkpoint (1 for Yes, 0 for No)
0
--CheckpointInterval
Epoch interval to save checkpoints
5

Example Command for Supervised Training
python train.py --BasePath ./Data --LogsPath ./Logs --NumEpochs 20 --MiniBatchSize 64 --ModelType Sup

Example Command for Unsupervised Training
python train.py --BasePath ./Data --LogsPath ./Logs --NumEpochs 15 --MiniBatchSize 32 --ModelType Unsup


TensorBoard Visualization
You can monitor the training progress using TensorBoard. Run the following command in the terminal:
tensorboard --logdir <path_to_logs_directory>

Open the provided URL in your browser to view the training and validation loss plots.

Directory Requirements
Ensure that the following directories are correctly structured before running:
 Data/
├── Train/          # Directory with original training images
└── Val/            # Directory with original validation images

Notes
Device Support: The scripts automatically utilize a CUDA-enabled GPU if available. Otherwise, they fall back to CPU.
Gradient Clipping: Enabled to prevent exploding gradients.
Learning Rate: Adjust the learning rate as needed in train.py (default: 0.0001).
Data Normalization: Images are normalized to [0, 1] for input to the network.
