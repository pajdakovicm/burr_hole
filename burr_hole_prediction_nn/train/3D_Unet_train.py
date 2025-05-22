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
)
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import Transform, MapTransform
from torch.optim.lr_scheduler import StepLR
from monai.networks.nets import UNet


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
            RandCropByPosNegLabeld(
                keys=["image", "hematoma", "label"],
                label_key="hematoma",  # Use hematoma mask for guidance
                spatial_size=(256, 256, 32),
                pos=1,  # Ensure hematoma is in patch
                neg=0,  # No pure background
                num_samples=2,
            ),
            lambda data: {
                "image": np.concatenate([data["image"], data["hematoma"]], axis=0),
                "label": data["label"],
            },
            # Crop around hematoma region (ensure burr hole is in patch)
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
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

    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
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
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    # start a typical PyTorch training
    val_interval = 2  # runs validation every 2 epochs
    best_metric = -1
    best_metric_epoch = -1  # stores the epoch number when the best dice was achieved
    epoch_loss_values = list()  # training loss per epoch
    metric_values = list()  # validation Dice scores per epoch
    writer = SummaryWriter("runs/unet3D")
    for epoch in range(70):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{70}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()  # clear gradients
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()  # compute gradients
            optimizer.step()  # update weigths
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}/{70}, Current Learning Rate: {current_lr}")

        # validation every 2 epochs
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
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
                    # val_epoch_len = len(val_check_ds)
                    val_loss = loss_function(val_outputs, val_labels)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_epoch_loss += val_loss.item()
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                print(
                    "current epoch: {} current val_loss: {:.4f}".format(
                        epoch + 1, val_epoch_loss / val_step
                    )
                )
                writer.add_scalar("val_loss", val_epoch_loss / val_step, epoch + 1)
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        "/mnt/bio/CT/burr_hole/chsdh_gt1/results/unet_best_metric_model_segmentation3d_array_2.pth",
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            #   plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


if __name__ == "__main__":
    temdir = "/mnt/bio/CT/burr_hole/chsdh_gt1"
    main(temdir)
