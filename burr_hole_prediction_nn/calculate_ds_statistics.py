import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm


def compute_ct_intensity_stats(data_dir, pattern="*.nii.gz"):
    """
    Computes mean, std, 0.5th percentile, and 99.5th percentile
    of intensity values in CT scans across a dataset.

    Args:
        data_dir (str): Path to the directory containing the CT scans.
        pattern (str): File pattern to match CT files (default: '*.nii.gz').

    Returns:
        dict: Dictionary with 'mean', 'std', 'percentile_0_5', 'percentile_99_5'
    """
    file_paths = sorted(glob(f"{data_dir}/**/{pattern}", recursive=True))
    all_voxels = []

    for path in tqdm(file_paths, desc="Processing CT scans"):
        img = nib.load(path).get_fdata()
        img = img[np.isfinite(img)]  # filter out NaNs/Infs just in case
        all_voxels.append(img.flatten())

    all_voxels = np.concatenate(all_voxels)

    stats = {
        "mean": np.mean(all_voxels),
        "std": np.std(all_voxels),
        "percentile_0_5": np.percentile(all_voxels, 0.5),
        "percentile_99_5": np.percentile(all_voxels, 99.5),
    }

    return stats


def compute_avg_ct_shape(data_dir, pattern="*.nii.gz"):
    """
    Computes the average shape (x, y, z) of CT volumes across the dataset.

    Args:
        data_dir (str): Path to the directory containing CT scans.
        pattern (str): File pattern to match CT files (default: '*.nii.gz').

    Returns:
        dict: Dictionary with average shape for 'x', 'y', and 'z' axes.
    """
    file_paths = sorted(glob(f"{data_dir}/**/{pattern}", recursive=True))
    shapes = []

    for path in tqdm(file_paths, desc="Measuring CT shapes"):
        img = nib.load(path)
        shape = img.shape
        if len(shape) == 3:
            shapes.append(shape)
        else:
            print(f"Skipping non-3D volume: {path}")

    shapes = np.array(shapes)
    avg_shape = np.mean(shapes, axis=0)

    return {"avg_x": avg_shape[0], "avg_y": avg_shape[1], "avg_z": avg_shape[2]}


if __name__ == "__main__":
    preop_path = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/burr_hole_training_data/gt1/preop"
    stats = compute_ct_intensity_stats(preop_path)
    avg_shape = compute_avg_ct_shape(preop_path)
    print(stats)
    print(avg_shape)
