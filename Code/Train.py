import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
# os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 backend
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)  # Prevent conflicting paths
import cv2
import random
import argparse
from tqdm import tqdm
from Network_ import HomographyModel  # Ensure this points to your modified network file
import datetime
import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys  # Import the sys module

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("##############################################################")
print(f"Using device: {device}")

def GenerateBatch(BasePath, DirNames, Labels, MiniBatchSize, is_training=True):
    """
    Generate batches for training or validation.
    """
    img_a_batch, img_b_batch, displacements = [], [], []

    split_dir = "Train" if is_training else "Val"

    for i in range(MiniBatchSize):
        # Randomly select an image
        idx = random.randint(0, len(DirNames) - 1)
        img_original_path = os.path.join(BasePath, "GeneratedData", split_dir, 'Original', DirNames[idx])
        img_warped_path = os.path.join(BasePath, "GeneratedData", split_dir, 'Warped', DirNames[idx])

        # Load and preprocess original and warped images
        img_a = cv2.imread(img_original_path, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(img_warped_path, cv2.IMREAD_GRAYSCALE)

        # Resize and normalize to [0, 1]
        img_a = cv2.resize(img_a, (128, 128)) / 255.0
        img_b = cv2.resize(img_b, (128, 128)) / 255.0

        # Expand dimensions for PyTorch compatibility (1 channel)
        img_a = np.expand_dims(img_a, axis=0)  # Shape: (1, 128, 128)
        img_b = np.expand_dims(img_b, axis=0)  # Shape: (1, 128, 128)

        # Load the corresponding ground-truth perturbation
        perturbation = Labels[idx]
        displacements.append(torch.tensor(perturbation.reshape(-1), dtype=torch.float32))

        # Append images to their respective batches
        img_a_batch.append(torch.tensor(img_a, dtype=torch.float32))
        img_b_batch.append(torch.tensor(img_b, dtype=torch.float32))

    return torch.stack(img_a_batch).to(device), torch.stack(img_b_batch).to(device), torch.stack(displacements).to(device)

def save_training_graphs(experiment_dir, train_losses, val_losses):
    """Saves loss curves as PNG and CSV in experiment directory"""
    # Create plots
    plt.figure(figsize=(12, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save plots
    plot_path = os.path.join(experiment_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save data as CSV
    df = pd.DataFrame({
        'Epoch': range(len(train_losses)),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses
    })
    csv_path = os.path.join(experiment_dir, 'training_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Loss metrics have been saved to {csv_path}")

def TrainOperation(BasePath, DirNamesTrain, TrainLabels, DirNamesVal, ValLabels, ModelType, **kwargs):
    """
    Train and validate the model.
    """
    # Create a unique experiment directory
    experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(kwargs['LogsPath'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(experiment_dir)

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Model and Optimizer Initialization
    model = HomographyModel(model_type=ModelType).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Load from checkpoint if specified
    start_epoch = 0
    if kwargs['LatestFile']:
        print(f"Loading checkpoint from {kwargs['LatestFile']}")  # Add this line
        checkpoint = torch.load(kwargs['LatestFile'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])  # Load previous train losses
        val_losses = checkpoint.get('val_losses', [])      # Load previous val losses
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("Starting training from scratch...")

    # Training Loop
    for epoch in tqdm(range(start_epoch, kwargs['NumEpochs']), desc="Epochs"):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        num_batches_train = len(DirNamesTrain) // kwargs['MiniBatchSize']

        for _ in tqdm(range(num_batches_train), desc="Training"):
            img_a, img_b, displacements = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, kwargs['MiniBatchSize'])

            if ModelType == 'Sup':
                batch = (img_a, img_b, displacements)
                loss = model.training_step(batch, None)
            else:  # Unsup
                corners_a = torch.tensor([[0, 0], [0, 128], [128, 0], [128, 128]], dtype=torch.float32).unsqueeze(0).repeat(kwargs['MiniBatchSize'], 1, 1).to(device)
                batch = (img_a, img_b, corners_a, img_a, img_b)  # For unsupervised photometric loss
                loss = model.training_step(batch, None)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_batches_train
        train_losses.append(avg_train_loss)  # Store train loss
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss}")

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        num_batches_val = len(DirNamesVal) // kwargs['MiniBatchSize']

        with torch.no_grad():
            for _ in tqdm(range(num_batches_val), desc="Validation"):
                img_a, img_b, displacements = GenerateBatch(BasePath, DirNamesVal, ValLabels, kwargs['MiniBatchSize'], is_training=False)

                if ModelType == 'Sup':
                    batch = (img_a, img_b, displacements)
                    loss = model.validation_step(batch, None)
                else:  # Unsup
                    corners_a = torch.tensor([[0, 0], [0, 128], [128, 0], [128, 128]], dtype=torch.float32).unsqueeze(0).repeat(kwargs['MiniBatchSize'], 1, 1).to(device)
                    batch = (img_a, img_b, corners_a, img_a, img_b)
                    loss = model.validation_step(batch, None)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / num_batches_val
        val_losses.append(avg_val_loss)  # Store val loss
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        print(f"Epoch {epoch}: Val Loss = {avg_val_loss}")

        # Save checkpoint
        if (epoch % kwargs['SaveCheckPoint']) == 0 or epoch == kwargs['NumEpochs'] -1:  # Save at interval or end
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.ckpt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,  # Save train losses
                'val_losses': val_losses       # Save val losses
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    print("Training finished!")

def TrainOperation1(BasePath, DirNamesTrain, TrainLabels, DirNamesVal, ValLabels, ModelType, **kwargs):
    """
    Train and validate the model with integrated loss visualization
    """

    model = HomographyModel(model_type=ModelType).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    torch.cuda.empty_cache()

    # Create experiment directory
    experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(kwargs['LogsPath'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize trackers
    writer = SummaryWriter(experiment_dir)
    train_losses, val_losses = [], []

    def save_training_graphs():
        """Save loss curves to experiment directory"""
        plt.figure(figsize=(12, 6))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Loss Curves (Best: {min(val_losses):.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Error bars
        plt.subplot(1, 2, 2)
        plt.errorbar(
            range(len(val_losses)),
            val_losses,
            yerr=np.std([val_losses[max(0,i-5):i+1] for i in range(len(val_losses))], axis=1),
            ecolor='lightgray',
            elinewidth=3
        )
        plt.title('Validation Loss with Standard Deviation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Save plots
        plot_path = os.path.join(experiment_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save CSV
        pd.DataFrame({
            'Epoch': range(len(train_losses)),
            'Train_Loss': train_losses,
            'Val_Loss': val_losses
        }).to_csv(os.path.join(experiment_dir, 'loss_data.csv'), index=False)
        print("Saved training metrics graph and data")

    for epoch in tqdm(range(kwargs['NumEpochs']), desc="Epochs"):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        num_batches_train = len(DirNamesTrain) // kwargs['MiniBatchSize']

        for _ in tqdm(range(num_batches_train), desc="Training"):
            img_a, img_b, displacements = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, kwargs['MiniBatchSize'])

            if ModelType == 'Sup':
                batch = (img_a, img_b, displacements)
                loss = model.training_step(batch, None)
            else:
                corners_a = torch.tensor([[0, 0], [0, 128], [128, 0], [128, 128]],
                                       dtype=torch.float32).unsqueeze(0).repeat(kwargs['MiniBatchSize'], 1, 1).to(device)
                batch = (img_a, img_b, corners_a, img_a, img_b)
                loss = model.training_step(batch, None)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_batches_train
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss}")

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        num_batches_val = len(DirNamesVal) // kwargs['MiniBatchSize']

        with torch.no_grad():
            for _ in tqdm(range(num_batches_val), desc="Validation"):
                img_a, img_b, displacements = GenerateBatch(BasePath, DirNamesVal, ValLabels,
                                                           kwargs['MiniBatchSize'], is_training=False)

                if ModelType == 'Sup':
                    batch = (img_a, img_b, displacements)
                    loss = model.validation_step(batch, None)
                else:
                    corners_a = torch.tensor([[0, 0], [0, 128], [128, 0], [128, 128]],
                                           dtype=torch.float32).unsqueeze(0).repeat(kwargs['MiniBatchSize'], 1, 1).to(device)
                    batch = (img_a, img_b, corners_a, img_a, img_b)
                    loss = model.validation_step(batch, None)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / num_batches_val
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        print(f"Epoch {epoch}: Val Loss = {avg_val_loss}")

        # Checkpoint and Visualization
        if (epoch % kwargs['SaveCheckPoint']) == 0:
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.ckpt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            save_training_graphs()  # Update visualizations

    # Final save after training
    save_training_graphs()
    print(f"\n📈 Final training graphs saved to: {experiment_dir}")
    writer.close()

def FindLatestModel(checkpoint_dir):
    """Locates most recent .ckpt checkpoint"""
    ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    return max(ckpts, key=os.path.getctime) if ckpts else None

def SetupAll(base_path, checkpoint_path):
    """Returns mock dataset configuration - implement actual logic per your data"""
    return (
        sorted(os.listdir(os.path.join(base_path, "GeneratedData/Train/Original"))),  # DirNamesTrain
        sorted(os.listdir(os.path.join(base_path, "GeneratedData/Val/Original"))),    # DirNamesVal
        1,                          # SaveCheckPoint: Save every 5 epochs
        (128, 128),                 # ImageSize
        5000,                       # NumTrainSamples
        1000,                       # NumValSamples
        None,                       # TrainCoordinates
        None,                       # ValCoordinates
        8,                          # NumClasses (8-DoF homography)
        None,                       # TrainCorners
        None                        # ValCorners
    )

def PrettyPrint(epochs, batch_size, samples, checkpoint):
    """Displays training configuration"""
    print(f"Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Training Samples: {samples}")
    print(f"Resuming from: {checkpoint if checkpoint else 'No checkpoint'}")

def main():
    # Redirect stdout to a log file
    log_file_path = "training_log.txt"
    sys.stdout = open(log_file_path, "w")
    print(f"All print statements are redirected to {log_file_path}")

    parser = argparse.ArgumentParser()
    # Add LoadCheckPoint argument for model resuming
    parser.add_argument("--LoadCheckPoint", type=int, default=0,
                      help="1 to load existing checkpoint, 0 to train from scratch")
    # Existing arguments remain unchanged
    parser.add_argument("--BasePath", default=r"/home/manideep/CV/p1p2/YourDirectoryID_p1/YourDirectoryID_p1/Phase2/Data", help="Path to the dataset")
    parser.add_argument("--LogsPath", default="Logs/", help="Path to save TensorBoard logs")
    parser.add_argument("--NumEpochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--MiniBatchSize", type=int, default=32, help="Batch size")
    parser.add_argument("--ModelType", choices=['Sup', 'Unsup'], default='Sup', help="Model type (Supervised or Unsupervised)")
    parser.add_argument("--CheckpointInterval", type=int, default=1, help="Epoch interval between checkpoint saves (default: 5)")

    args = parser.parse_args()

    print("\n====== Initializing Training Setup ======")
    # Setup all configuration parameters
    (
        DirNamesTrain,        # Training image filenames
        DirNamesVal,          # Validation image filenames
        SaveCheckPoint,       # Save checkpoint every N epochs
        ImageSize,            # Input image dimensions
        NumTrainSamples,      # Total training samples
        NumValSamples,        # Total validation samples
        TrainCoordinates,     # Training coordinates data
        ValCoordinates,       # Validation coordinates data
        NumClasses,           # Number of output classes
        TrainCorners,         # Training image corners
        ValCorners,           # Validation image corners
    ) = SetupAll(args.BasePath, args.LogsPath)  # Initialize dataset configuration

    # Checkpoint handling
    LatestFile = None
    if args.LoadCheckPoint == 1:
        print("\n🔍 Searching for existing checkpoints...")
        LatestFile = FindLatestModel(args.LogsPath)
        if LatestFile:
            print(f"✅ Found existing checkpoint: {LatestFile}")
        else:
            print("⚠️  No checkpoints found, starting fresh training")

    # Display training configuration
    print("\n📊 Training Parameters:")
    PrettyPrint(args.NumEpochs, args.MiniBatchSize, NumTrainSamples, LatestFile)

    print("\n🚀 Starting Training Process...")
    # Existing data loading remains unchanged
    train_dir = os.path.join(args.BasePath, "GeneratedData", "Train")
    val_dir = os.path.join(args.BasePath, "GeneratedData", "Val")
    train_labels = np.load(os.path.join(train_dir, "Labels", "perturbations.npy"))
    # val_labels = np.zeros((1000, 8))
    val_labels = np.load(os.path.join(val_dir, "Labels", "perturbations.npy"))

    # Original TrainOperation call with additional parameters
    TrainOperation(
        BasePath=args.BasePath,
        DirNamesTrain=DirNamesTrain,
        TrainLabels=train_labels,
        DirNamesVal=DirNamesVal,
        ValLabels=val_labels,
        ModelType=args.ModelType,
        LogsPath=args.LogsPath,
        NumEpochs=args.NumEpochs,
        MiniBatchSize=args.MiniBatchSize,
        # Additional parameters from SetupAll
        SaveCheckPoint=SaveCheckPoint,
        ImageSize=ImageSize,
        NumTrainSamples=NumTrainSamples,
        NumValSamples=NumValSamples,
        TrainCoordinates=TrainCoordinates,
        ValCoordinates=ValCoordinates,
        NumClasses=NumClasses,
        TrainCorners=TrainCorners,
        ValCorners=ValCorners,
        LatestFile=LatestFile
    )

    # Restore stdout and close log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Training completed. All print statements are saved to {log_file_path}")

if __name__ == "__main__":
    main()
