import SimpleITK as sitk
import os
import argparse


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

    parser = argparse.ArgumentParser(
        description="Subtract the registered and preoperative image."
    )
    parser.add_argument(
        "--input_dir_registered",
        type=str,
        required=True,
        help="Path to the directory containing registered images .",
    )
    parser.add_argument(
        "--input_dir_preop",
        type=str,
        required=True,
        help="Path to the directory containing preoperative images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where subtracted images will be saved.",
    )

    args = parser.parse_args()
    subtract_images(
        input_dir_preop=args.input_dir_preop,
        input_dir_registered=args.input_dir,
        output_dir=args.output_dir_subtracted,
    )
