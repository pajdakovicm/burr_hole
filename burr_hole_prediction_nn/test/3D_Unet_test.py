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
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
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
)

# from Unet3D_train import CTNormalizationd
from monai.transforms import Transform, MapTransform, ConcatItemsd


class CTNormalizationd(MapTransform):
    def __init__(self, keys, intensity_properties, target_dtype=np.float32):
        """
        初始化CTNormalization转换。
        :param keys: 字典中要转换的键列表
        :param intensity_properties: 包含强度相关属性的字典（均值、标准差、百分位数边界等）
        :param target_dtype: 转换目标的数据类型
        """
        super().__init__(keys)
        self.intensity_properties = intensity_properties
        self.target_dtype = target_dtype

    def __call__(self, data):
        """
        在图像上应用CT标准化。
        :param data: 包含图像数据的字典
        :return: 包含标准化图像数据的字典
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


def main(tempdir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = sorted(glob(os.path.join(tempdir, "test/image", "*.nii.gz")))
    hems = sorted(glob(os.path.join(tempdir, "test/hematoma", "*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "test/label", "*.nii.gz")))
    val_files = [
        {"image": img, "hematoma": hem, "label": seg}
        for img, hem, seg in zip(images, hems, segs)
    ]

    # define transforms for image and segmentation
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
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate
    )
    val_check_data = monai.utils.misc.first(val_loader)
    print(val_check_data["image"].shape, val_check_data["label"].shape)

    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(
        output_dir="/mnt/bio/CT/burr_hole/chsdh_gt1/predictions",
        output_ext=".nii.gz",
        output_postfix="seg",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "/mnt/bio/CT/burr_hole/chsdh_gt1/results/unet_best_metric_model_segmentation3d_array_2.pth"
        )
    )
    model.eval()
    number = 0
    # output_directory = "/mnt/bio/CT/burr_hole/chsdh_gt1/predictions"
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(
                device
            )
            # define sliding window size and batch size for windows inference
            roi_size = (256, 256, 16)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, model
            )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            image_tensor = val_data["image"][0]  # a MetaTensor if loaded correctly
            image_meta = image_tensor.meta
            print("image meta is this")
            print(image_meta)
            original_affine = image_meta["affine"]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            iou_metric(y_pred=val_outputs, y=val_labels)
            hd_metric(y_pred=val_outputs, y=val_labels)
            number += 1
            name = f"{number}.nii.gz"
            # nib.save(
            #     nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
            #     os.path.join(output_directory, name),
            # )
            for val_output in val_outputs:
                saver(val_output)
        # aggregate the final mean dice result
        print("evaluation dice metric:", dice_metric.aggregate().item())
        print("evaluation iou metric:", iou_metric.aggregate().item())
        print("evaluation hd metric:", hd_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    temdir = "/mnt/bio/CT/burr_hole/chsdh_gt1"
    main(temdir)
