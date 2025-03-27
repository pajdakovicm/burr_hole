import numpy as np
from scipy.signal import correlate
import SimpleITK as sitk
import os
from tqdm import tqdm
import argparse


def template_matching(image, template):
    """
    Performs template matching using cross-correlation.
    Args:
        image (numpy.ndarray): The input image in NumPy array format.
        template (numpy.ndarray): The template to match in the image.

    Returns:
        numpy.ndarray: The correlation map, where higher values indicate better matches.
    """
    correlation_map = correlate(
        image, template, mode="same", method="auto"
    )  # keep size same
    return correlation_map


# used for testing
def match_template_one_image(image_path, template_path, output_path, threshold):
    """
    Applies template matching to a single image and saves the result.
    Args:
        image_path (str): Path to the input NIfTI image.
        template_path (str): Path to the template NIfTI image.
        output_path (str): Path where the output masked image will be saved.
        threshold (float): Threshold value for filtering matches (range: 0 to 1).

    Returns:
        None
    """
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    template = sitk.ReadImage(template_path)
    template_array = sitk.GetArrayFromImage(template)
    print(f"Image shape: {image_array.shape}")
    print(f"Template shape: {template_array.shape}")
    print(f"Image spacing: {image.GetSpacing()}")
    print(f"Template spacing: {template.GetSpacing()}")
    # Compute the correlation map
    correlation_map = template_matching(image_array, template_array)
    # correlation_map = match_template(image_array, template_array)
    correlation_map = (correlation_map - correlation_map.min()) / (
        correlation_map.max() - correlation_map.min()
    )
    binary_map = correlation_map >= threshold
    masked_array = image_array * binary_map
    binary_mask = (masked_array > 0).astype(np.uint8)
    masked_image = sitk.GetImageFromArray(binary_mask)
    masked_image.CopyInformation(image)
    sitk.WriteImage(masked_image, output_path)


def template_matching_dataset(input_dir, template_path, output_dir, threshold):
    """
    Applies template matching to all images in a dataset.
    Args:
        input_dir (str): Path to the directory containing input NIfTI images.
        template_path (str): Path to the template NIfTI image.
        output_dir (str): Path where output masked images will be saved.
        threshold (float): Threshold for filtering template matches (range: 0 to 1).

    Returns:
        None
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file '{template_path}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]

    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in '{input_dir}'.")

    try:
        template = sitk.ReadImage(template_path)
        template_array = sitk.GetArrayFromImage(template)
    except Exception as e:
        raise RuntimeError(f"Error loading template image '{template_path}': {e}")

    for file in tqdm(files, desc="Matching input images with template..", unit="file"):
        try:
            image_path = os.path.join(input_dir, file)
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)

            correlation_map = template_matching(image_array, template_array)
            # normalization of correlation map
            correlation_map = (correlation_map - correlation_map.min()) / (
                correlation_map.max() - correlation_map.min()
            )
            # apply threshold
            binary_map = correlation_map >= threshold
            masked_array = image_array * binary_map
            binary_mask = (masked_array > 0).astype(np.uint8)

            masked_image = sitk.GetImageFromArray(binary_mask)
            masked_image.CopyInformation(image)  # preserve spatial metadata

            output_name = file.replace("subtracted", "label")
            output_path = os.path.join(output_dir, output_name)
            sitk.WriteImage(masked_image, output_path)

        except Exception as e:
            print(f"Error processing file '{file}': {e}")
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply template matching.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing .nii.gz files.",
    )
    parser.add_argument(
        "--template", required=True, help="Path to the template NIfTI file."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory where images with matched template will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for filtering template matches of correlation map (default: 0.7).",
    )
    args = parser.parse_args()

    template_matching_dataset(
        input_dir=args.input_dir,
        template_path=args.template,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
