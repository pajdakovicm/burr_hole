import os
import SimpleITK as sitk
from tqdm import tqdm
import argparse

### resample dataset


def resample_volume(volume_path, interpolator, new_spacing):
    """
    Resamples a image to a new voxel spacing.
    Args:
        volume_path (str): Path to the input medical image file.
        interpolator (sitk.InterpolatorEnum, optional): Interpolation method for resampling.
            Default is `sitk.sitkLinear`.
        new_spacing (list of float, optional): Target voxel spacing in the format `[x, y, z]`.

    Returns:
        sitk.Image: The resampled 3D image with the new spacing.
    """
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32)  # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    return sitk.Resample(
        volume,
        new_size,
        sitk.Transform(),
        interpolator,
        volume.GetOrigin(),
        new_spacing,
        volume.GetDirection(),
        0,
        volume.GetPixelID(),
    )


def resample_dataset(
    input_dir, output_dir, new_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkLinear
):
    """
    Resample all images in the input directory and save them to the output directory.

    Parameters:
    - input_dir: Path to the directory containing the input images.
    - output_dir: Path to the directory to save the resampled images.
    - target_spacing: tuple of floats (e.g., (1.0, 1.0, 1.0))
    - interpolator: Interpolation method (default: sitk.sitkLinear)

    returns: None.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]

    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in '{input_dir}'.")

    for file_name in tqdm(files, desc="Resampling images..", unit="file_name"):
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            resampled_image = resample_volume(
                input_path, interpolator=interpolator, new_spacing=new_spacing
            )
            sitk.WriteImage(resampled_image, output_path)


def get_interpolator(interpolator_name):
    """
    Convert string interpolator name to SimpleITK interpolator.

    Args:
        interpolator_name (str).

    Returns:
        sitk.InterpolatorEnum.
    """
    interpolators = {
        "NearestNeighbor": sitk.sitkNearestNeighbor,
        "Linear": sitk.sitkLinear,
        "BSpline": sitk.sitkBSpline,
        "Gaussian": sitk.sitkGaussian,
    }
    return interpolators.get(interpolator_name, sitk.sitkLinear)  # default linear


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Resample images to a new voxel spacing."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing .nii.gz files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for resampled images.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="New voxel spacing as three space-separated values (e.g., -s 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--interpolator",
        type=str,
        choices=[
            "NearestNeighbor",
            "Linear",
            "BSpline",
            "Gaussian",
        ],
        default="Linear",
        help="Interpolation method (default: linear).",
    )

    args = parser.parse_args()
    interpolator = get_interpolator(args.interpolator)

    resample_dataset(
        args.input_dir,
        args.output_dir,
        args.spacing,
        interpolator,
    )
