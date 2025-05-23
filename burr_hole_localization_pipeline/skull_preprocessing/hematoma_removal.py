import os
import nibabel as nib
from cut_skull import save_nifty
import SimpleITK as sitk
import argparse


def clear_skull(input_dir, threshold, output_dir):
    """
    Excludes unwanted regions (hematoma) of the CT images by applying a threshold.

    Args:
        input_dir (str): Path to the directory containing input NIfTI images.
        threshold (float): Intensity threshold to distinguish skull from other tissues.
        output_dir (str): Path to the directory where segmented skull images will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith("nii.gz"):
            file_path = os.path.join(input_dir, file)
            file_name = os.path.basename(file_path)
            nii_file = nib.load(file_path)
            img_data = nii_file.get_fdata()
            skull_mask = img_data > threshold
            img_data[~skull_mask] = 1
            # apply mask to original image
            segmented_skull = img_data * skull_mask
            save_nifty(
                segmented_skull,
                os.path.join(output_dir, file_name),
                nii_file.affine,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Skull clearing and image subtraction pipeline."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing input images (with hematoma).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100,
        help="Intensity HU threshold to remove hematoma.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where images without hematoma will be saved.",
    )

    args = parser.parse_args()
    # clear the skull
    clear_skull(
        input_dir=args.input_dir,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
