import logging
import os, random
import sys
import tempfile
from glob import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import (
    ImageDataset,
    create_test_image_3d,
    decollate_batch,
    DataLoader,
    pad_list_data_collate,
    list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    Spacingd,
    Orientationd,
    RandCropByPosNegLabel,
    LoadImaged,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
)
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import Transform, MapTransform
from torch.optim.lr_scheduler import StepLR
from monai.networks.nets import UNet
from torch.nn.functional import mse_loss


def make_deterministic(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


make_deterministic(42)


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes Pearson correlation coefficient between two tensors.
    Both should be 1D or flattened.
    """
    x = x.flatten()
    y = y.flatten()

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    return (numerator / denominator).item() if denominator != 0 else 0.0


class CTNormalizationd(MapTransform):
    def __init__(self, keys, intensity_properties, target_dtype=np.float32):
        """
        CTNormalization.
        :param keys:
        :param intensity_properties:
        :param target_dtype:
        """
        super().__init__(keys)
        self.intensity_properties = intensity_properties
        self.target_dtype = target_dtype

    def __call__(self, data):
        """
        :param data:
        :return:
        """
        d = dict(data)
        for key in self.keys:
            assert (
                self.intensity_properties is not None
            ), "CTNormalizationd requires intensity properties"
            d[key] = d[key].astype(self.target_dtype)
            mean_intensity = self.intensity_properties["mean"]
            std_intensity = self.intensity_properties["std"]
            lower_bound = self.intensity_properties["percentile_00_5"]
            upper_bound = self.intensity_properties["percentile_99_5"]
            d[key] = np.clip(d[key], lower_bound, upper_bound)
            d[key] = (d[key] - mean_intensity) / max(std_intensity, 1e-8)
        return d


from monai.transforms import Transform, MapTransform

import numpy as np
from scipy.ndimage import label
import torch


def binary_mask_to_heatmap(binary_mask, sigma=3.0, spacing=(1.0, 1.0, 1.0)):
    """
    Convert 3D binary mask to Gaussian heatmap with spacing-aware sigma
    Args:
        binary_mask: 3D numpy array (D, H, W)
        sigma: physical space standard deviation in mm
        spacing: voxel spacing (z, y, x) in mm
    Returns:
        heatmap: 3D numpy array with Gaussian blobs
    """
    # Convert physical sigma to voxel space
    voxel_sigmas = [sigma / s for s in spacing]

    labeled_mask, num_features = label(binary_mask)
    heatmap = np.zeros_like(binary_mask, dtype=np.float32)

    zz, yy, xx = np.indices(binary_mask.shape)

    for i in range(1, num_features + 1):
        positions = np.argwhere(labeled_mask == i)
        if len(positions) == 0:
            continue

        # Find centroid in physical coordinates
        centroid_voxel = np.mean(positions, axis=0)
        z_cent, y_cent, x_cent = centroid_voxel

        # Calculate anisotropic Gaussian
        z_dist = (zz - z_cent) ** 2 / (2 * voxel_sigmas[0] ** 2)
        y_dist = (yy - y_cent) ** 2 / (2 * voxel_sigmas[1] ** 2)
        x_dist = (xx - x_cent) ** 2 / (2 * voxel_sigmas[2] ** 2)

        gaussian = np.exp(-(z_dist + y_dist + x_dist))
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap


from monai.transforms import Transform, MapTransform


class BinaryToHeatmapd(MapTransform):
    def __init__(self, keys, sigma=5.0, spacing=(1.0, 1.0, 1.0)):
        """
        Convert binary mask to heatmap with fixed spacing
        Args:
            sigma: physical size (mm) of Gaussian kernel
            spacing: manual voxel spacing in (z, y, x) order
        """
        super().__init__(keys)
        self.sigma = sigma
        self.spacing = spacing  # Fixed manual spacing

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Remove channel dimension and convert to numpy
            binary_mask = (x
                d[key][0].cpu().numpy()
                if isinstance(d[key], torch.Tensor)
                else d[key][0]
            )

            # Generate heatmap with manual spacing
            heatmap = binary_mask_to_heatmap(
                binary_mask, sigma=self.sigma, spacing=self.spacing
            )

            # Add channel dimension and update data
            d[key] = heatmap[np.newaxis]  # (C, D, H, W)

        return d


def main(tempdir):

    # Load preoperative CTs, hematoma masks, and burr hole labels
    images = sorted(glob(os.path.join(tempdir, "train/image", "*.nii.gz")))
    hems = sorted(
        glob(os.path.join(tempdir, "train/hematoma", "*.nii.gz"))
    )  # Hematoma masks
    burr_labels = sorted(
        glob(os.path.join(tempdir, "train/label", "*.nii.gz"))
    )  # Heatmaps or coordinates

    val_images = sorted(glob(os.path.join(tempdir, "val/image", "*.nii.gz")))
    val_hems = sorted(glob(os.path.join(tempdir, "val/hematoma", "*.nii.gz")))
    val_burr_labels = sorted(glob(os.path.join(tempdir, "val/label", "*.nii.gz")))

    # MONAI data format (include hematoma as input)
    train_files = [
        {"image": img, "hematoma": hem, "label": burr}
        for img, hem, burr in zip(images, hems, burr_labels)
    ]
    val_files = [
        {"image": img, "hematoma": hem, "label": burr}
        for img, hem, burr in zip(val_images, val_hems, val_burr_labels)
    ]

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "hematoma", "label"]),
            EnsureChannelFirstd(keys=["image", "hematoma", "label"]),
            EnsureTyped(keys=["image", "hematoma", "label"]),
            Orientationd(keys=["image", "hematoma", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "hematoma", "label"],
                pixdim=(
                    0.43692008,
                    0.43692014,
                    2.8643477,
                ),  # datasets mean across each dimension
                mode=("bilinear", "nearest", "nearest"),  # Hematoma and label are masks
            ),
            CTNormalizationd(
                keys=["image"],
                intensity_properties={
                    "mean": 90.53903379712949,
                    "std": 33.3241868209507,  # datasets mean std and percentiles
                    "percentile_00_5": 80.0,
                    "percentile_99_5": 200.0,
                },
            ),
            # Convert coordinates to heatmap (uncomment if labels are coordinates)
            # BurrHoleToHeatmapD(keys=["label"], sigma=3),
            # Concatenate CT and hematoma into 2-channel input
            # Crop around hematoma region (ensure burr hole is in patch)
            BinaryToHeatmapd(
                keys=["label"],
                sigma=5.0,
                spacing=[
                    0.43692008,
                    0.43692014,
                    2.8643477,
                ],
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label", "hematoma"],
                label_key="label",
                spatial_size=(256, 256, 16),
                pos=3,
                neg=1,
                num_samples=4,
            ),
            #
            RandFlipd(keys=["image", "label", "hematoma"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label", "hematoma"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label", "hematoma"], prob=0.5, spatial_axis=2),
            lambda data: {
                "image": np.concatenate([data["image"], data["hematoma"]], axis=0),
                "label": data["label"],
            },
            RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=0.5),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "hematoma", "label"]),
            EnsureChannelFirstd(keys=["image", "hematoma", "label"]),
            EnsureTyped(keys=["image", "hematoma", "label"]),
            Orientationd(keys=["image", "hematoma", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "hematoma", "label"],
                pixdim=(0.43692008, 0.43692014, 2.8643477),
                mode=("bilinear", "nearest", "nearest"),
            ),
            CTNormalizationd(
                keys=["image"],
                intensity_properties={
                    "mean": 90.53903379712949,
                    "std": 33.3241868209507,  # datasets mean std and percentiles
                    "percentile_00_5": 80.0,
                    "percentile_99_5": 200.0,
                },
            ),
            BinaryToHeatmapd(
                keys=["label"],
                sigma=5.0,
                spacing=[
                    0.43692008,
                    0.43692014,
                    2.8643477,
                ],
            ),
            lambda data: {
                "image": np.concatenate([data["image"], data["hematoma"]], axis=0),
                "label": data["label"],
            },
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transform)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,  # loads data in batches of 4
        shuffle=True,
        num_workers=1,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate
    )

    train_check_data = monai.utils.misc.first(train_loader)
    print(train_check_data["image"].shape, train_check_data["label"].shape)

    val_check_data = monai.utils.misc.first(val_loader)
    print(val_check_data["image"].shape, val_check_data["label"].shape)

    # dice_metric = DiceMetric(
    #    include_background=True, reduction="mean", get_not_nans=False
    # )
    # post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),  # downsampling
        num_res_units=2,
    ).to(device)
    loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    # start a typical PyTorch training
    max_epochs = 50
    val_interval = 2
    best_val_mse = float("inf")
    best_epoch = -1
    epoch_loss_values = []
    writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(
                device
            )

            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            writer.add_scalar(
                "train_loss", loss.item(), epoch * len(train_loader) + step
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validation every val_interval epochs
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_images, val_labels = val_data["image"].to(device), val_data[
                        "label"
                    ].to(device)
                    roi_size = (256, 256, 16)
                    sw_batch_size = 16
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                    )
                    val_loss = loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()

                avg_val_mse = val_epoch_loss / val_step
                print(f"Epoch {epoch + 1} - Avg. Validation MSE: {avg_val_mse:.4f}")
                writer.add_scalar("val_loss", avg_val_mse, epoch + 1)

                if avg_val_mse < best_val_mse:
                    best_val_mse = avg_val_mse
                    best_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        "/mnt/bio/CT/burr_hole/chsdh_gt1/results/unet_best_mse_model.pth",
                    )
                    print("Saved new best model (based on MSE)")

                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")

            # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"Training completed. Best MSE: {best_val_mse:.4f} at epoch {best_epoch}")
    writer.close()


if __name__ == "__main__":
    temdir = "/mnt/bio/CT/burr_hole/chsdh_gt1"
    main(temdir)
