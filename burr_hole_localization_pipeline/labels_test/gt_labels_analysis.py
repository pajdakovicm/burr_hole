import nibabel as nib
import numpy as np
import pandas as pd
import os
import matplotlib as plt
import matplotlib.pyplot as plt


# used for testing and debugging
def filter_area_range(csv_path, min_value=900, max_value=1200):
    """
    This function reads a CSV file containing region properties, filters rows where
    the "area" column falls within the specified range, and returns the filtered DataFrame.
    Args:
        csv_path (str): Path to the CSV file containing region properties.
        min_value (int, optional): Minimum area threshold for filtering (default: 900).
        max_value (int, optional): Maximum area threshold for filtering (default: 1200).
    Returns:
        pandas.DataFrame: A DataFrame containing only the rows where the area is within the specified range.
    """
    df = pd.read_csv(csv_path)
    filtered_df = df[(df["area"] > min_value) & (df["area"] < max_value)]
    return filtered_df


def plot_area_distribution(csv_path, column_name="area"):
    """
    Plots a histogram of the specified column to visualize its distribution.
    Args:
        csv_path (str): Path to the CSV file.
        column_name (str): Name of the column containing the area values.

    Returns:
        None.
    """
    # load the dataset
    df = pd.read_csv(csv_path)
    # compute mean and standard deviation
    mean_value = df[column_name].mean()
    std_value = df[column_name].std()

    plt.figure(figsize=(8, 5))
    plt.hist(df[column_name], bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(mean_value, color="red", linestyle="dashed", linewidth=2, label="Mean")
    plt.axvline(
        mean_value + std_value,
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label="Mean + 1 Std",
    )
    plt.axvline(
        mean_value - std_value,
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label="Mean - 1 Std",
    )
    # labels and title
    plt.xlabel("Volume")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Volume Across Ground Truth Lables.")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()


def compute_labels_upper_percentage_dataset(image_dir, label_dir):
    """
    Computes the percentage of labeled voxels that are positioned in the upper 50% of the skull
    for all 'postop' images in the dataset.
    Args:
        image_dir (str): Directory containing postoperative NIfTI images.
        label_dir (str): Directory containing corresponding label NIfTI images.

    Returns:
        float: Overall percentage of labeled voxels in the upper skull across the dataset.
    """
    total_labels = 0
    total_labels_in_upper_half = 0

    for file in os.listdir(image_dir):
        if file.endswith(".nii.gz") and "postop" in file.lower():
            skull_path = os.path.join(image_dir, file)
            label_name = file.replace("postop", "mask")
            label_path = os.path.join(label_dir, label_name)

            if not os.path.exists(label_path):
                print(f"Skipping {file}: No corresponding label file found.")
                continue

            skull_img = nib.load(skull_path)
            skull_data = skull_img.get_fdata()

            label_img = nib.load(label_path)
            label_data = label_img.get_fdata()
            skull_voxels = np.argwhere(skull_data > 0)
            if skull_voxels.size == 0:
                print(f"Skipping {file}: No skull found.")
                continue  # if no skull is found

            # compute skull midpoint (z-axis)
            min_z, max_z = np.min(skull_voxels[:, 2]), np.max(skull_voxels[:, 2])
            mid_z = (min_z + max_z) / 2
            #  mid_z = 0.5 * skull_voxels[:, 2]
            label_voxels = np.argwhere(label_data > 0)
            labels_in_upper_half = label_voxels[:, 2] > mid_z
            total_labels += len(label_voxels)
            total_labels_in_upper_half += np.sum(labels_in_upper_half)

            percentage_upper = np.mean(labels_in_upper_half) * 100
            print(f"{file}: {percentage_upper:.2f}% of labels in upper skull.")

    # compute overall percentage for the dataset
    overall_percentage = (
        (total_labels_in_upper_half / total_labels) * 100 if total_labels > 0 else 0
    )
    print(f"\nOverall Percentage of Labels in Upper Skull: {overall_percentage:.2f}%")
    return overall_percentage


def check_hole_hu_values(image_path):
    """
    Computes the HU values within the burr hole region (blob) in a subtracted image (template in temolate matching).
    Args:
        image_path (str): Path to the subtracted NIfTI image.
        hole_mask_path (str): Path to the binary mask of the burr hole region.

    Returns:
        None
    """
    # Load the subtracted image (contains HU values)
    image = nib.load(image_path)
    image_data = image.get_fdata()
    # Print statistics
    print(f"Total hole voxels: {image_data.size}")
    print(f"Min HU in hole: {np.min(image_data)}")
    print(f"Max HU in hole: {np.max(image_data)}")
    print(f"Mean HU in hole: {np.mean(image_data)}")
    print(f"Median HU in hole: {np.median(image_data)}")
    print(f"Standard deviation: {np.std(image_data)}")


if __name__ == "__main__":

    # plot the distribution of area of blobs in the ground truth dataset, resampled
    csv_path = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/csv_files/gt_resampled_regionprops.csv"
    plot_area_distribution(csv_path=csv_path, column_name="area")
    # skull_path = "/Users/marijapajdakovic/Desktop/burr_hole_project/data/chSDH_dataset/18_postop.nii.gz"
    # label_path = "/Users/marijapajdakovic/Desktop/burr_hole_project/data/gt_labels/18_label.nii.gz"
    # image_dir = "/Users/marijapajdakovic/Desktop/burr_hole_project/data/chSDH_dataset/"
    # label_dir = "/Users/marijapajdakovic/Desktop/burr_hole_project/data/gt_labels"

    # check HU values of chosen template blob used in template matching
    hole_region_path = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/templates/18_template.nii.gz"
    check_hole_hu_values(image_path=hole_region_path)
