import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops_table
import pandas as pd
import os
import SimpleITK as sitk
from skimage.measure import regionprops_table, label
from tqdm import tqdm
import argparse


# this script excludes false positive blobs with the utilization of region properties filtering.


def find_connected_components(binary_image_sitk, connectivity=3):
    """
    Find connected components in 3D binary image.
    Args:
        binary_image_sitk: SimpleITK binary image.
        connectivity: Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (1-3 for 3D)
    Returns:
        labeled_image_sitk: SimpleITK image with labeled components
        num_components: Total number of connected components
    """
    binary_array = sitk.GetArrayFromImage(binary_image_sitk)
    # connected component labeling
    labeled_array, num_components = label(
        binary_array, connectivity=connectivity, return_num=True
    )
    labeled_image_sitk = sitk.GetImageFromArray(labeled_array.astype(np.uint16))
    labeled_image_sitk.CopyInformation(binary_image_sitk)
    return labeled_image_sitk, num_components


def compute_aspect_ratios(region):
    """
    Computes aspect ratios from bounding box dimensions.
    Args:
        region (skimage.measure._regionprops.RegionProperties).
    Returns:
        tuple: (max/min ratio, mid/min ratio)
    """
    minr, minc, mind, maxr, maxc, maxd = region.bbox
    depth = maxr - minr
    height = maxc - minc
    width = maxd - mind
    sizes = sorted([depth, height, width])
    max_ratio = sizes[2] / sizes[0] if sizes[0] > 0 else 1
    mid_ratio = sizes[1] / sizes[0] if sizes[0] > 0 else 1
    return max_ratio, mid_ratio


def region_props(
    image_path,
    output_dir,
    csv_filename,
    area_min,
    area_max,
    aspect_ratio_max_min,
    aspect_ratio_mid_min,
    solidity_threshold,
):
    """
    Computes region properties for connected components in a 3D image. It filters components based onf area,
    aspect ratio and solidity. It saves the results to .csv file.
    Args:
        image_path (str): Path to the input NIfTI image.
        output_dir (str): Path to save the output binary mask and CSV file.
        csv_filename (str): Name of the CSV file where extracted properties are stored.
        area_min (int): Minimum area threshold for filtering components.
        area_max (int): Maximum area threshold for filtering components.
        aspect_ratio_max_min (float): Maximum allowed ratio of max/min bounding box dimensions.
        aspect_ratio_mid_min (float): Maximum allowed ratio of mid/min bounding box dimensions.

    Returns:
        None
    """

    image = sitk.ReadImage(image_path, sitk.sitkUInt8)
    labeled_image, num_components = find_connected_components(image, connectivity=3)
    print(f"Found {num_components} connected components in {image_path}")

    props = regionprops_table(
        sitk.GetArrayFromImage(labeled_image),
        properties=(
            "label",
            "area",
            "centroid",
            "bbox",
            "equivalent_diameter_area",
            "solidity",
        ),
    )

    df = pd.DataFrame(props)
    # extract dimensions to compute aspect ratios
    df["depth"] = df["bbox-3"] - df["bbox-0"]
    df["height"] = df["bbox-4"] - df["bbox-1"]
    df["width"] = df["bbox-5"] - df["bbox-2"]
    sizes = df[["depth", "height", "width"]].values
    min_dim = np.min(sizes, axis=1)
    mid_dim = np.median(sizes, axis=1)
    max_dim = np.max(sizes, axis=1)
    df["max_min_aspect_ratio"] = max_dim / min_dim
    df["mid_min_aspect_ratio"] = mid_dim / min_dim

    output_name = os.path.basename(image_path)
    os.makedirs(output_dir, exist_ok=True)
    # append data to the global csv file, for debugging
    df["image_name"] = output_name
    csv_path = os.path.join(output_dir, csv_filename)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    print(f"Appended region properties for {output_name} to {csv_path}")

    # filter only properties that fullfill the conditions
    selected_labels = df.loc[
        (df["area"] < area_max)
        & (df["area"] > area_min)
        & (df["max_min_aspect_ratio"] < aspect_ratio_max_min)
        & (df["mid_min_aspect_ratio"] < aspect_ratio_mid_min)
        & (df["solidity"] >= solidity_threshold),
        "label",
    ].values

    # create a binary mask with only the selected components
    labeled_np = sitk.GetArrayFromImage(labeled_image)
    binary_mask = np.isin(labeled_np, selected_labels).astype(np.uint8)
    # save image
    binary_sitk_image = sitk.GetImageFromArray(binary_mask)
    binary_sitk_image.CopyInformation(image)
    output_img_path = os.path.join(output_dir, output_name)
    sitk.WriteImage(binary_sitk_image, output_img_path)
    print(f"Filtered image saved to {output_img_path}")


def region_props_over_dataset(
    input_dir,
    output_dir,
    area_min,
    area_max,
    csv_filename,
    aspect_ratio_max_min,
    aspect_ratio_mid_min,
    solidity_threshold,
):
    """
    Computes region properties for all images in a dataset with progress tracking and error handling.

    Args:
        input_dir (str): Directory containing input NIfTI images.
        output_dir (str): Directory where results will be saved.
        area_min (int): Minimum area threshold for filtering components.
        area_max (int): Maximum area threshold for filtering components.
        csv_filename (str): Name of the CSV file where results are stored.
        aspect_ratio_max_min (float): Maximum allowed ratio of max/min bounding box dimensions.
        aspect_ratio_mid_min (float): Maximum allowed ratio of mid/min bounding box dimensions.
        solidity_threshold (float): Threshold for solidity filtering.

    Returns:
        None
    """

    # error checks
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    os.makedirs(output_dir, exist_ok=True)
    # Validate numerical parameters
    if not (isinstance(area_min, (int, float)) and area_min >= 0):
        raise ValueError(
            f"Invalid area_min: {area_min}. Must be a non-negative number."
        )
    if not (isinstance(area_max, (int, float)) and area_max > area_min):
        raise ValueError(
            f"Invalid area_max: {area_max}. Must be greater than area_min."
        )
    if not (0 <= aspect_ratio_max_min and 0 <= aspect_ratio_mid_min):
        raise ValueError("Aspect ratios must be non-negative.")
    if not (0 <= solidity_threshold <= 1):
        raise ValueError("Solidity threshold must be between 0 and 1.")

    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in '{input_dir}'.")

    for file in tqdm(files, desc="Extracting only wanted regions..", unit="file"):
        image_path = os.path.join(input_dir, file)

        try:
            region_props(
                image_path=image_path,
                output_dir=output_dir,
                csv_filename=csv_filename,
                area_min=area_min,
                area_max=area_max,
                aspect_ratio_max_min=aspect_ratio_max_min,
                aspect_ratio_mid_min=aspect_ratio_mid_min,
                solidity_threshold=solidity_threshold,
            )
        except Exception as e:
            print(f"Skipping file {file} due to an error {e}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute region properties for CT images."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing .nii.gz files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for processed images.",
    )
    parser.add_argument(
        "--area_min",
        type=int,
        required=True,
        help="Minimum area threshold for filtering.",
    )
    parser.add_argument(
        "--area_max",
        type=int,
        required=True,
        help="Maximum area threshold for filtering.",
    )
    parser.add_argument(
        "--csv_filename",
        required=True,
        help="Path to the CSV file to store region properties information.",
    )
    parser.add_argument(
        "--aspect_ratio_max_min",
        type=float,
        required=True,
        help="Max allowed ratio of max/min bounding box dimensions.",
    )
    parser.add_argument(
        "--aspect_ratio_mid_min",
        type=float,
        required=True,
        help="Max allowed ratio of mid/min bounding box dimensions.",
    )
    parser.add_argument(
        "--solidity_threshold",
        type=float,
        required=True,
        help="Threshold for solidity filtering (0 to 1).",
    )

    args = parser.parse_args()

    region_props_over_dataset(
        args.input_dir,
        args.output_dir,
        args.area_min,
        args.area_max,
        args.csv_filename,
        args.aspect_ratio_max_min,
        args.aspect_ratio_mid_min,
        args.solidity_threshold,
    )
