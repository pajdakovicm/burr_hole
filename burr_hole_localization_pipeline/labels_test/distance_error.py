import numpy as np
from scipy.ndimage import center_of_mass
import os
import SimpleITK as sitk
import csv
import argparse
import matplotlib.pyplot as plt

import os
import numpy as np
import csv
import SimpleITK as sitk
from scipy.ndimage import center_of_mass


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
    try:
        pred_center = center_of_mass(pred_label)
        gt_center = center_of_mass(gt_label)

        if any(np.isnan(pred_center)) or any(np.isnan(gt_center)):
            print("Warning: NaN detected during calculation of center.")
            return float("nan")

        distance_voxels = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
        distance_mm = distance_voxels * np.array(voxel_size)

        total_distance_mm = np.sqrt(np.sum(np.square(distance_mm)))

        return total_distance_mm
    except Exception as e:
        print(f"Error calculating Euclidean distance: {e}")
        return float("nan")


def compute_distance_error_over_dataset(gt_dir, pred_dir, output_csv):
    """
    Computes the Euclidean distance error for all NIfTI images in a dataset.
    Args:
        gt_dir (str): Directory containing ground truth NIfTI files.
        pred_dir (str): Directory containing predicted NIfTI files.
        output_csv (str): Name of the output CSV file storing distance errors.
    Returns:
        list, float, float: List of errors, mean error, and standard deviation.
    """
    errors_list = []
    results = []

    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith(".nii.gz"):
            try:
                gt_path = os.path.join(gt_dir, gt_file)
                prefix_number = extract_prefix(os.path.basename(gt_path))
                pred_file = f"{prefix_number}_label.nii.gz"
                pred_path = os.path.join(pred_dir, pred_file)

                if os.path.exists(pred_path):
                    gt_image = sitk.ReadImage(gt_path)
                    pred_image = sitk.ReadImage(pred_path)
                    voxel_size = gt_image.GetSpacing()

                    gt_array = sitk.GetArrayFromImage(gt_image) > 0
                    pred_array = sitk.GetArrayFromImage(pred_image) > 0

                    error = calculate_euclidean_distance(
                        gt_array, pred_array, voxel_size
                    )

                    if not np.isnan(error):
                        errors_list.append(error)
                        results.append([gt_file, error])
                    else:
                        print(f"Warning: NaN error for file {gt_file}.")
                else:
                    print(f"Warning: Prediction file not found for {gt_file}.")
            except Exception as e:
                print(f"Error processing file {gt_file}: {e}")

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Euclidean Distance Error (mm)"])
        writer.writerows(results)
    print(f"Results saved to {output_csv}")

    # mean_error = np.mean(errors_list) if errors_list else float("nan")
    # std_error = np.std(errors_list) if errors_list else float("nan")
    # return errors_list, mean_error, std_error
    median_error = np.median(errors_list) if errors_list else float("nan")
    iqr_error = (
        np.percentile(errors_list, 75) - np.percentile(errors_list, 25)
        if errors_list
        else float("nan")
    )
    return errors_list, median_error, iqr_error


# plot mean and std
def plot_distribution_mean(data, title="Data Distribution"):
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

    # plot median


def plot_distribution_median(
    errors_list, title="Distribution of Burr Hole Localization Error"
):
    """
    Plots the distribution of errors with the median.

    Args:
        errors_list (list): List of error values.
        title (str): Title of the plot.
    """
    median_line = np.median(errors_list)

    plt.figure(figsize=(8, 5))
    plt.hist(
        errors_list, bins=30, color="blue", alpha=0.6, edgecolor="black", density=False
    )

    plt.axvline(
        median_line, color="orange", linestyle="--", label=f"Median: {median_line:.2f}"
    )

    plt.xlabel("Distance (mm)")
    plt.ylabel("Frequency")
    plt.title(title)
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
    errors_list, median_error, iqr_error = compute_distance_error_over_dataset(
        args.gt_dir, args.pred_dir, args.output_csv
    )

    print(f"Median Euclidean Distance Error: {median_error} mm")
    print(f"Interquartile Range (IQR) of Euclidean Distance Error: {iqr_error} mm")

    print(f"Errors for each image are saved in: {args.output_csv}")
    plot_distribution_median(errors_list)
    plot_distribution_mean(errors_list)
