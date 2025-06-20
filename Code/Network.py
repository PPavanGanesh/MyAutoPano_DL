import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia

# Loss function for both supervised and unsupervised cases
def LossFn(predicted, target, loss_type='MSE'):
    if loss_type == 'MSE':
        return torch.nn.MSELoss()(predicted, target)
    elif loss_type == 'L1':
        return torch.nn.L1Loss()(predicted, target)
    else:
        raise ValueError("Invalid loss_type. Choose 'MSE' or 'L1'.")


class SupModel(nn.Module):
    def __init__(self, InputSize=2, OutputSize=8):
        """
        CNN model for predicting 8 corner displacements given concatenated original and warped patches.
        """
        super(SupModel, self).__init__()
        self.conv1 = nn.Conv2d(InputSize, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, OutputSize)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)

        x = self.dropout(x)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TensorDLT(nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()

    def forward(self, corners_a, preds):
        """
        Compute the homography matrix using Direct Linear Transform (DLT) given corner displacements.
        """
        batch_size = corners_a.size(0)
        corners_a = corners_a.view(batch_size, 4, 2)
        preds = preds.view(batch_size, 4, 2)

        # Formulate matrices for solving Ax = b
        A = []
        for i in range(4):
            x, y = corners_a[:, i, 0], corners_a[:, i, 1]
            u, v = preds[:, i, 0], preds[:, i, 1]
            A.append(torch.stack([-x, -y, -torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(y), torch.zeros_like(x), u * x, u * y], dim=1))
            A.append(torch.stack([torch.zeros_like(x), torch.zeros_like(y), torch.zeros_like(x), -x, -y, -torch.ones_like(x), v * x, v * y], dim=1))
        A = torch.cat(A, dim=1).view(batch_size, 8, 8)
        b = preds.view(batch_size, 8, 1)

        # Solve for x in Ax = b using least-squares
        H = torch.linalg.pinv(A) @ b
        H = torch.cat([H, torch.ones((batch_size, 1, 1), device=H.device)], dim=1)  # Add [0, 0, 1]
        return H.view(-1, 3, 3)


class UnSupModel(pl.LightningModule):
    def __init__(self):
        super(UnSupModel, self).__init__()
        self.feature_extractor = SupModel()
        self.tensor_dlt = TensorDLT()

    def forward(self, patch_a, patch_b, corners_a, img_a):
        """
        Forward pass of the unsupervised model. Computes the photometric loss using TensorDLT.
        """
        # Step 1: Predict displacements
        input_batch = torch.cat((patch_a, patch_b), dim=1)
        predicted_displacements = self.feature_extractor(input_batch)

        # Step 2: Estimate homography using TensorDLT
        H = self.tensor_dlt(corners_a, predicted_displacements)

        # Step 3: Warp image_a using the homography
        warped_img_a = kornia.geometry.transform.warp_perspective(img_a, H, dsize=(128, 128))

        return warped_img_a, predicted_displacements

    def compute_photometric_loss(self, warped_img, img_b):
        """
        Compute the photometric loss between the warped original image and the target.
        """
        return F.l1_loss(warped_img, img_b)

    def training_step(self, batch, batch_idx):
        patch_a, patch_b, corners_a, img_a, img_b = batch

        # Forward pass
        warped_img_a, _ = self.forward(patch_a, patch_b, corners_a, img_a)

        # Compute photometric loss
        loss = self.compute_photometric_loss(warped_img_a, img_b)
        self.log('train_loss_unsup', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class HomographyModel(pl.LightningModule):
    def __init__(self, model_type="Sup"):
        """
        Initialize supervised or unsupervised homography models.
        """
        super(HomographyModel, self).__init__()
        self.model_type = model_type

        if model_type == "Sup":
            self.model = SupModel()
        elif model_type == "Unsup":
            self.model = UnSupModel()
        else:
            raise ValueError("Invalid model type. Choose 'Sup' or 'Unsup'.")

    def forward(self, *inputs):
        return self.model(*inputs)

    def training_step(self, batch, batch_idx):
        if self.model_type == "Sup":
            original_patches, warped_patches, ground_truth_displacements = batch
            input_batch = torch.cat((original_patches, warped_patches), dim=1)
            predicted_displacements = self.model(input_batch)
            loss = LossFn(predicted_displacements, ground_truth_displacements)
            self.log('train_loss', loss)
        elif self.model_type == "Unsup":
            loss = self.model.training_step(batch, batch_idx)  # Calls unsupervised training step
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_type == "Sup":
            original_patches, warped_patches, ground_truth_displacements = batch
            input_batch = torch.cat((original_patches, warped_patches), dim=1)
            predicted_displacements = self.model(input_batch)
            loss = LossFn(predicted_displacements, ground_truth_displacements)
            self.log('val_loss', loss)
        elif self.model_type == "Unsup":
            patch_a, patch_b, corners_a, img_a, img_b = batch
            warped_img_a, _ = self.model.forward(patch_a, patch_b, corners_a, img_a)
            loss = self.model.compute_photometric_loss(warped_img_a, img_b)
            self.log('val_loss_unsup', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
