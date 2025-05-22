import logging, random
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch, monai

from monai import config
from monai.data import (
    ImageDataset,
    create_test_image_3d,
    decollate_batch,
    DataLoader,
    list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Spacingd,
    Activations,
    LoadImaged,
    AsDiscrete,
    Compose,
    SaveImage,
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    Lambda,
)

# Add the required transforms from training script
from monai.transforms import MapTransform
from torch.nn.functional import mse_loss
from monai.data.meta_tensor import MetaTensor

from monai.transforms import ConcatItemsd, LambdaD


def debug_concat(data):
    print("\n[DEBUG] Keys in batch:", list(data.keys()))

    if "image" not in data or "hematoma" not in data:
        raise KeyError("Missing 'image' or 'hematoma' in the data dictionary")

    img = data["image"]
    hem = data["hematoma"]

    print(f"[DEBUG] image: shape={getattr(img, 'shape', None)}, type={type(img)}")
    print(f"[DEBUG] hematoma: shape={getattr(hem, 'shape', None)}, type={type(hem)}")

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"image is not a Tensor, got {type(img)}")
    if not isinstance(hem, torch.Tensor):
        raise TypeError(f"hematoma is not a Tensor, got {type(hem)}")

    if img.shape[1:] != hem.shape[1:]:
        raise ValueError(
            f"Shape mismatch: image shape {img.shape}, hematoma shape {hem.shape}"
        )

    data["image"] = torch.cat([img, hem], dim=0)
    return data


# Include the custom transforms from training
class CTNormalizationd(MapTransform):
    def __init__(self, keys, intensity_properties, target_dtype=np.float32):
        super().__init__(keys)
        self.intensity_properties = intensity_properties
        self.target_dtype = target_dtype

    def __call__(self, data):
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


class BinaryToHeatmapd(MapTransform):
    def __init__(self, keys, sigma=5.0, spacing=(1.0, 1.0, 1.0)):
        super().__init__(keys)
        self.sigma = sigma
        self.spacing = spacing

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            binary_mask = (
                d[key][0].cpu().numpy()
                if isinstance(d[key], torch.Tensor)
                else d[key][0]
            )
            heatmap = binary_mask_to_heatmap(
                binary_mask, sigma=self.sigma, spacing=self.spacing
            )
            d[key] = heatmap[np.newaxis]
        return d


def make_deterministic(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


make_deterministic(42)


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten()
    y = y.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    return (numerator / denominator).item() if denominator != 0 else 0.0


def main(tempdir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Load test data with hematoma masks
    images = sorted(glob(os.path.join(tempdir, "test/image", "*.nii.gz")))
    hems = sorted(glob(os.path.join(tempdir, "test/hematoma", "*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "test/label", "*.nii.gz")))
    test_files = [
        {"image": img, "hematoma": hem, "label": seg}
        for img, hem, seg in zip(images, hems, segs)
    ]

    # Update validation transform to match training
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "hematoma", "label"]),
            EnsureChannelFirstd(keys=["image", "hematoma", "label"]),
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
                    "std": 33.3241868209507,
                    "percentile_00_5": 80.0,
                    "percentile_99_5": 200.0,
                },
            ),
            EnsureTyped(
                keys=["image", "hematoma", "label"],
                track_meta=True,
            ),
            # LambdaD(
            #     keys=["image", "hematoma"],
            #     func=lambda img, hem: (
            #         print(f"Image shape: {img.shape}, Hematoma shape: {hem.shape}"),
            #         (img, hem),
            #     )[1],
            # ),
            ConcatItemsd(keys=["image", "hematoma"], name="image", dim=0),
            BinaryToHeatmapd(
                keys=["label"],
                sigma=5.0,
                spacing=[0.43692008, 0.43692014, 2.8643477],
            ),
        ]
    )

    test_ds = monai.data.Dataset(data=test_files, transform=val_transform)
    test_loader = DataLoader(
        test_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate
    )
    # saver = SaveImage(
    #     output_dir="./unet_output",
    #     output_ext=".nii.gz",
    #     output_postfix="pred",
    #     resample=False,
    #     separate_folder=False,
    #     meta_keys="image",  # tells it to use test_data["image"] for affine/meta
    #     output_dtype=np.float32,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=2,  # Changed from 1 to 2 for two input channels
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(
        torch.load("/mnt/bio/CT/burr_hole/chsdh_gt1/results/unet_best_mse_model.pth")
    )
    model.eval()

    total_mse = 0.0
    total_pearson = 0.0
    count = 0
    number = 0
    with torch.no_grad():
        output_directory = "/mnt/bio/CT/burr_hole/chsdh_gt1/predictions"
        for i, batch in enumerate(test_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            # image_meta = batch["image"].meta
            print(type(batch["image"][0]))

            image_tensor = batch["image"][0]  # a MetaTensor if loaded correctly
            image_meta = image_tensor.meta
            print("image meta is this")
            print(image_meta)
            original_affine = image_meta["affine"]
            # img_name = os.path.basename(image_meta["filename_or_obj"])
            # original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            #  print("Inference on case {}".format(img_name))

            roi_size = (256, 256, 16)
            sw_batch_size = 16
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)

            # Calculate metrics
            mse = mse_loss(outputs, labels).item()
            pearson = pearson_corr(outputs, labels)

            total_mse += mse
            total_pearson += pearson
            count += 1
            number += 1
            name = f"{number}.nii.gz"
            nib.save(
                nib.Nifti1Image(outputs.astype(np.uint8), original_affine),
                os.path.join(output_directory, name),
            )

    avg_mse = total_mse / count
    avg_pearson = total_pearson / count
    print(f"Test Results:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average Pearson Correlation: {avg_pearson:.4f}")


if __name__ == "__main__":
    temdir = "/mnt/bio/CT/burr_hole/chsdh_gt1"
    main(temdir)
