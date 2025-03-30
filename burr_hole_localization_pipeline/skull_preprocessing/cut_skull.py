import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import argparse


## This script cuts the lower parts of the skull from subtracted images obtained in registration.


def save_nifty(img_np, output_path, affine):
    """
    Saves a NIfTI image (.nii.gz).
    Args:
        img_np (numpy.ndarray): The image or binary mask to be saved.
        output_path (str): File path where the NIfTI image will be saved.
        affine (numpy.ndarray): 4x4 affine transformation matrix for spatial alignment.

    Returns:
        None
    """
    img_np = img_np.astype(np.int32)
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, output_path)


def cut_skull(input_dir, output_dir, cutoff):
    """
    Removes the lower portion of the skull in CT scans by setting it to zero.

    Args:
        input_dir (str): Path to the input directory with CT scans.
        output_dir (str): Path to the directory where images will be saved.
        cutoff (float): Fraction of the slices to be removed from the bottom  (z axis) (range 0 to 1).

    Returns:
        None
    """
    # error checks
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)
    if not (0 <= cutoff <= 1):
        raise ValueError(
            f"Invalid cutoff value: {cutoff}. It should be between 0 and 1."
        )

    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]

    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in '{input_dir}'.")

    for file in tqdm(files, desc="Cutting the skull..", unit="file"):
        try:
            input_path = os.path.join(input_dir, file)
            img = nib.load(input_path)
            img_data_sub = img.get_fdata()
            # cut lower parts of the skull
            num_slices = img_data_sub.shape[2]
            lower_bound = int(num_slices * cutoff)
            img_data_sub[:, :, :lower_bound] = 0
            output_path = os.path.join(output_dir, file)
            save_nifty(img_np=img_data_sub, output_path=output_path, affine=img.affine)
        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Cut lower portion of the skull in CT scans."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory should be the directory with images subtracted.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output directory for processed images.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="Fraction (0 to 1) of slices to be removed from the bottom (z-axis).",
    )

    args = parser.parse_args()
    cut_skull(args.input_dir, args.output_dir, args.cutoff)
