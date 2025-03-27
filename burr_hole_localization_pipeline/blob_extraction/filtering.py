import SimpleITK as sitk
import os
import SimpleITK as sitk
from tqdm import tqdm
import argparse


### Apply filter on dataset.


def filter_one_image(input_image_path, output_image_path):
    """
    Applies Curvature Anisotropic Diffusion filter.

    Args:
        input_image_path (str): Path to the input NIfTI image.
        output_image_path (str): Path where the filtered NIfTI image will be saved.

    Returns:
        None
    """
    image = sitk.ReadImage(input_image_path)
    smoothed_image = sitk.CurvatureAnisotropicDiffusion(
        sitk.Cast(image, sitk.sitkFloat32),
        timeStep=0.0625,
        conductanceParameter=3.0,
        numberOfIterations=7,
    )
    sitk.WriteImage(smoothed_image, output_image_path)


def filter_dataset(input_dir, output_dir):
    """
    Applies Curvature Anisotropic Diffusion filtering to all NIfTI images in a directory.

    Args:
        input_dir (str): Path to the directory containing the input images.
        output_dir (str): Path to the directory where filtered images will be saved.

    Returns: None
    """
    # error handling
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in '{input_dir}'.")
    for file in tqdm(
        files, desc="Applying Anisotropic Diffusion filter..", unit="file"
    ):
        image_path = os.path.join(input_dir, file)
        output_image_path = os.path.join(output_dir, file)
        try:
            filter_one_image(image_path, output_image_path)
            print(f"Successfully processed: {file}")
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply Anisotropic Diffusion filter to images."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory containing input files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory where filtered images will be saved.",
    )
    args = parser.parse_args()
    try:
        filter_dataset(args.input_dir, args.output_dir)
        print("Processing complete.")
    except Exception as e:
        print(f"Error: {e}")
