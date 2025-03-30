import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import argparse


def clip_ct(ct_numpy, min, max):
    """
    Clips the CT scan intensity values within a predefined range.
    Args:
        ct_numpy (numpy.ndarray): The CT scan represented as a NumPy array.
        min (int or float): The minimum intensity value allowed.
        max (int or float): The maximum intensity value allowed.

    Returns:
        numpy.ndarray: The clipped CT scan with intensity values within the specified range.
    """
    return ct_numpy.clip(min, max)


def save_nifty(img_np, output_path, affine):
    """
    Saves a NumPy array as a NIfTI (.nii.gz) file.

    Args:
        img_np (numpy.ndarray): The image or binary mask to be saved.
        output_path (str): The path where the NIfTI file will be saved (without extension).
        affine (numpy.ndarray): The affine transformation matrix for spatial metadata.

    Returns:
        None
    """
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, output_path + ".nii.gz")


def load_data(data_dir_path):
    """
    Loads and sorts preoperative and postoperative CT scan file paths.

    This function scans the given directory for files containing 'preop' and 'postop'
    in their names, sorts them numerically, and returns them as separate lists.

    Args:
        data_dir_path (str): The path to the directory containing CT scan files.

    Returns:
        tuple: A tuple containing two sorted NumPy arrays:
            - preop (numpy.ndarray): Sorted file paths of preoperative scans.
            - postop (numpy.ndarray): Sorted file paths of postoperative scans.
    """
    preop = np.array(
        [
            os.path.join(data_dir_path, scan)
            for scan in os.listdir(data_dir_path)
            if "preop" in scan
        ]
    )
    postop = np.array(
        [
            os.path.join(data_dir_path, scan)
            for scan in os.listdir(data_dir_path)
            if "postop" in scan
        ]
    )

    preop_indices = np.argsort(preop)
    postop_indices = np.argsort(postop)

    sorted_preop = preop[preop_indices]
    sorted_postop = postop[postop_indices]

    return sorted_preop, sorted_postop


def extract_prefix(string):
    """
    Extracts the prefix (first part) of a filename before the first underscore.

    Args:
        string (str): The input filename string.

    Returns:
        str: The extracted prefix.
    """
    parts = string.split("_")
    return parts[0]


def process_dataset(input_dir, output_dir, min_clip=-200, max_clip=1000):
    """
    Processes a dataset of CT scans by applying intensity clipping and saving the results.

    Args:
        input_dir (str): Path to the directory containing the preoperative and postoperative CT scans.
        output_dir (str): Path to the directory where the processed CT scans will be saved.
        min_clip (int, optional): Minimum Hounsfield Unit (HU) for window clipping.
        max_clip (int, optional): Maximum Hounsfield Unit (HU) for window clipping.

    Returns:
        None
    """
    preop, postop = load_data(input_dir)

    os.makedirs(output_dir, exist_ok=True)

    for preop_data, postop_data in tqdm(zip(preop, postop), desc="Processing CT Scans"):

        if not os.path.exists(preop_data) or not os.path.exists(postop_data):
            print(f"Skipping: Missing file(s) for {preop_data} or {postop_data}")
            continue

        try:
            preop_ct = nib.load(preop_data)
            postop_ct = nib.load(postop_data)

            preop_np = preop_ct.get_fdata()
            postop_np = postop_ct.get_fdata()

            preop_clipped = clip_ct(preop_np, min=min_clip, max=max_clip)
            postop_clipped = clip_ct(postop_np, min=min_clip, max=max_clip)

            preop_filename = f"{extract_prefix(os.path.basename(preop_data))}_preop"
            postop_filename = f"{extract_prefix(os.path.basename(postop_data))}_postop"

            save_nifty(
                preop_clipped, os.path.join(output_dir, preop_filename), preop_ct.affine
            )
            save_nifty(
                postop_clipped,
                os.path.join(output_dir, postop_filename),
                postop_ct.affine,
            )

        except Exception as e:
            print(f"Error processing {preop_data} and {postop_data}: {e}")
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess CT images and clip intensities."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory path to the initial (original) dataset of preoperative and postoperative CT scans.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory path to save the results."
    )
    parser.add_argument(
        "--min_clip",
        type=int,
        default=-200,
        help="Minimum HU value for clipping (default: -200).",
    )
    parser.add_argument(
        "--max_clip",
        type=int,
        default=1000,
        help="Maximum HU value for clipping (default: 1000).",
    )

    args = parser.parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
    )
