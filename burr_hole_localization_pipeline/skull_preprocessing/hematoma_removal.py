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


def subtract_images(input_dir_preop, input_dir_registered, output_dir):
    """
    Subtracts registered images from preoperative images.
    Args:
        input_dir_preop (str): Directory containing preoperative images.
        input_dir_registered (str): Directory containing registered images.
        output_dir (str): Directory to save subtracted images.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    preop = os.listdir(input_dir_preop)
    # include only preoperative images from dataset
    preop_files = [file for file in preop if "preop" in file]
    for preop_image_path in preop_files:
        registered_image_path = preop_image_path.replace("preop", "registered")
        # extract prefix number
        number = preop_image_path.split("_")[0]
        full_preop_path = os.path.join(input_dir_preop, preop_image_path)
        full_registered_path = os.path.join(input_dir_registered, registered_image_path)

        if os.path.exists(full_registered_path):
            try:
                preop_image = sitk.ReadImage(full_preop_path)
                registered_image = sitk.ReadImage(full_registered_path)
                difference_image = sitk.Subtract(preop_image, registered_image)
                # connvert to numpy array
                difference_array = sitk.GetArrayFromImage(difference_image)
                # check errors
                if difference_array is not None:
                    output_image = sitk.GetImageFromArray(difference_array)
                    output_image.SetSpacing(preop_image.GetSpacing())
                    output_image.SetOrigin(preop_image.GetOrigin())
                    output_image.SetDirection(preop_image.GetDirection())

                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, number + "_subtracted.nii.gz"
                    )
                    sitk.WriteImage(output_image, output_path)
                    print(f"Saved subtracted image: {output_path}")
                else:
                    print(f"Error in subraction {preop_image_path}")
            except Exception as e:
                print(f"Error processing {preop_image_path}: {e}")
                continue
        else:
            print(f"Registered image not found for {preop_image_path}")


if __name__ == "__main__":

    # input_dir = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/registered_no_hematoma"
    # # Save the cropped image
    # output_dir_no_hematoma = (
    #     "/Users/marijapajdakovic/Desktop/burr_hole_project/results/dataset_no_hematoma"
    # )
    # output_dir_subtracted = (
    #     "/Users/marijapajdakovic/Desktop/burr_hole_project/results/subtractions"
    # )

    # # clear_skull(input_dir=input_dir, output_dir=output_dir_no_hematoma, threshold=100)
    # subtract_images(
    #     input_dir_preop=output_dir_no_hematoma,
    #     input_dir_registered=input_dir,
    #     output_dir=output_dir_subtracted,

    # )
    parser = argparse.ArgumentParser(
        description="Skull clearing and image subtraction pipeline."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing registered images without hematoma.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100,
        help="Intensity threshold to distinguish skull from other tissues.",
    )
    parser.add_argument(
        "--output_dir_no_hematoma",
        type=str,
        required=True,
        help="Directory where images without hematoma will be saved.",
    )
    parser.add_argument(
        "--output_dir_subtracted",
        type=str,
        required=True,
        help="Directory where subtracted images will be saved.",
    )

    args = parser.parse_args()

    # First step: Clear skull
    clear_skull(
        input_dir=args.input_dir,
        threshold=args.threshold,
        output_dir=args.output_dir_no_hematoma,
    )

    # Second step: Subtract images
    subtract_images(
        input_dir_preop=args.output_dir_no_hematoma,
        input_dir_registered=args.input_dir,
        output_dir=args.output_dir_subtracted,
    )
