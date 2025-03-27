import SimpleITK as sitk
import os
import argparse

## transform gt labels to space of preoperative scans using registration transforms


# function to extract the first number
def extract_prefix(filename):
    return filename.split("_")[0]


# Process each ground truth label
def gt_to_preop_dataset(
    gt_labels_dir, transforms_dir, reference_images_dir, output_dir
):
    """
    Applies registration transformations to ground truth (GT) labels to align them with preoperative images.
    Args:
        gt_labels_dir (str): Path to the directory containing ground truth label images (.nii.gz).
        transforms_dir (str): Path to the directory containing transformation files (.tfm).
        reference_images_dir (str): Path to the directory containing reference registered images (.nii.gz).
        output_dir (str): Path to the directory where transformed labels will be saved.
    Returns:
        None
    """
    for gt_file in os.listdir(gt_labels_dir):
        if gt_file.endswith(".nii.gz"):
            prefix = extract_prefix(gt_file)
            # file paths
            transformation_file = os.path.join(
                transforms_dir, f"{prefix}_registered.tfm"
            )
            reference_image_file = os.path.join(
                reference_images_dir, f"{prefix}_registered.nii.gz"
            )

            if os.path.exists(transformation_file) and os.path.exists(
                reference_image_file
            ):
                print(f"Processing: {gt_file}")

                # load the binary label, transformation, and reference image
                gt_label_path = os.path.join(gt_labels_dir, gt_file)
                label_image = sitk.ReadImage(gt_label_path, sitk.sitkUInt8)
                transform = sitk.ReadTransform(transformation_file)
                reference_image = sitk.ReadImage(reference_image_file)

                # resample the binary label
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(reference_image)
                resampler.SetTransform(transform)
                resampler.SetInterpolator(
                    sitk.sitkNearestNeighbor
                )  # set nearest neighbor for binary labels
                resampler.SetOutputPixelType(sitk.sitkUInt8)
                transformed_label = resampler.Execute(label_image)

                output_path = os.path.join(output_dir, f"{prefix}_label.nii.gz")
                sitk.WriteImage(transformed_label, output_path)
                print(f"Saved transformed label to: {output_path}")
            else:
                print(f"Missing transformation or reference image for: {gt_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply registration transformations to ground truth labels to align them with preoperative images."
    )

    parser.add_argument(
        "--gt_labels_dir",
        required=True,
        help="Path to the directory containing ground truth label images (.nii.gz).",
    )
    parser.add_argument(
        "--transforms_dir",
        required=True,
        help="Path to the directory containing transformation files (.tfm).",
    )
    parser.add_argument(
        "--reference_images_dir",
        required=True,
        help="Path to the directory containing registered images (.nii.gz).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory where transformed labels will be saved.",
    )

    args = parser.parse_args()

    gt_to_preop_dataset(
        args.gt_labels_dir,
        args.transforms_dir,
        args.reference_images_dir,
        args.output_dir,
    )
