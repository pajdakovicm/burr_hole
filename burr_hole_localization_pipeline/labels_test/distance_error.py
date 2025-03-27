import numpy as np
from scipy.ndimage import center_of_mass
import os
import SimpleITK as sitk
import csv
import argparse
import matplotlib.pyplot as plt


def plot_distribution(data, title="Data Distribution"):
    """
    Plots the distribution of the given data with its mean and standard deviation.

    Parameters:
        data (array-like): The dataset to plot.
        title (str): Title of the plot.
    """
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)

    # plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, color="blue", alpha=0.6, edgecolor="black", density=True)

    # add vertical lines for mean and standard deviation
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
    plt.axvline(
        mean + std, color="green", linestyle="--", label=f"Mean + 1σ: {mean + std:.2f}"
    )
    plt.axvline(
        mean - std, color="green", linestyle="--", label=f"Mean - 1σ: {mean - std:.2f}"
    )

    plt.xlabel("Distance (mm)")
    plt.ylabel("Frequency")
    plt.title("The Distribution of Burr Hole Localization Error")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def extract_prefix(string):
    """
    Extracts the prefix (first part) of a filename before the first underscore.
    Args:
        string (str): The input filename string.
    Returns:
        str: The extracted prefix.

    """
    parts = string.split("_")
    prefix = parts[0]
    return prefix


def calculate_euclidean_distance(pred_label, gt_label, voxel_size):
    """
    This function calculates the center of mass for both the predicted and ground truth labels,
    computes the Euclidean distance in voxel space, and then converts it to millimeters using
    the provided voxel size.
    Args:
        pred_label (numpy.ndarray): Binary numpy array representing the predicted label.
        gt_label (numpy.ndarray): Binary numpy array representing the ground truth label.
        voxel_size (tuple of float): Voxel size in millimeters, given as (x, y, z) dimensions.

    Returns:
        float: The Euclidean distance between the two centers of mass.
    """
    pred_center = center_of_mass(pred_label)
    gt_center = center_of_mass(gt_label)
    distance_voxels = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
    distance_mm = distance_voxels * np.array(voxel_size)

    total_distance_mm = np.sqrt(np.sum(np.square(distance_mm)))  # to mm

    return total_distance_mm


def compute_distance_error_over_dataset(gt_dir, pred_dir, output_csv):
    """
    Computes the Dice coefficient for all NIfTI images in a dataset.
    Args:
        gt_dir (str): Directory containing ground truth NIfTI files.
        pred_dir (str): Directory containing predicted NIfTI files.
        output_csv (str, optional): Name of the output CSV file storing Dice scores.
    Returns:
        float: The mean Dice coefficient over the dataset.
    """

    errors_list = []
    # used to append information to csv file
    results = []
    # for every gile in gt dataset
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".nii.gz"):
            gt_path = os.path.join(gt_dir, gt_file)

            # find coresponding prediction
            prefix_number = extract_prefix(os.path.basename(gt_path))
            pred_file = f"{prefix_number}_label.nii.gz"
            pred_path = os.path.join(pred_dir, pred_file)

            if os.path.exists(pred_path):
                # load images
                gt_image = sitk.ReadImage(gt_path)
                pred_image = sitk.ReadImage(pred_path)
                voxel_size = gt_image.GetSpacing()

                gt_array = sitk.GetArrayFromImage(gt_image) > 0
                pred_array = sitk.GetArrayFromImage(pred_image) > 0

                error = calculate_euclidean_distance(
                    gt_array, pred_array, voxel_size=voxel_size
                )
                # append error
                errors_list.append(error)

                results.append([gt_file, error])
    # write to csv file, for debugging
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Euclidean Distance Error (mm)"])
        writer.writerows(results)

    return errors_list, np.mean(errors_list), np.std(errors_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute Euclidean distance errors over a dataset (Distance between the center of mass of GT and automatically obtained label)."
    )

    parser.add_argument(
        "--gt_dir",
        required=True,
        help="Directory containing ground truth files.",
    )

    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing automatically obtained label files.",
    )

    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to the output CSV file where the errors will be saved.",
    )

    args = parser.parse_args()

    # Call the function with parsed arguments
    errors_list, mean_error, std_error = compute_distance_error_over_dataset(
        args.gt_dir, args.pred_dir, args.output_csv
    )

    # Optionally, print out some statistics
    print(f"Mean Euclidean Distance Error: {mean_error} mm")
    print(f"Standard Deviation of Euclidean Distance Error: {std_error} mm")
    print(f"Errors for each image are saved in: {args.output_csv}")
