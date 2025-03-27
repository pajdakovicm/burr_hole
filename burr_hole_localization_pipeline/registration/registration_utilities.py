import numpy as np
import SimpleITK as sitk
import os


def save_transform_and_image(
    transform,
    fixed_image,
    moving_image,
    outputfile_prefix_transform,
    outputfile_prefix_image,
):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid and save the
    result to file.

    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
        outputfile_prefix (string): transform is written to outputfile_prefix.tfm and resampled image is written to
                                    outputfile_prefix.mha.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)

    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)
    sitk.WriteImage(resample.Execute(moving_image), outputfile_prefix_image + ".nii.gz")
    sitk.WriteTransform(transform, outputfile_prefix_transform + ".tfm")
    return resample.Execute(moving_image)


def load_data(data_dir_path):
    """
    Load and sort medical scan file paths from a specified directory.
    Parameters:
    data_dir_path (str): The path to the directory containing the scan files.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - sorted_preop (np.array): An array of sorted file paths for pre-operative scans.
        - sorted_postop (np.array): An array of sorted file paths for post-operative scans.
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
    return preop, postop


def extract_prefix(string):
    """
    Extracts the prefix (first part) of a given string before the first underscore.

    Args:
        string (str): The input string, typically a filename or identifier.

    Returns:
        str: The extracted prefix, which is the substring before the first underscore.

    Example:
        >>> extract_prefix("patient123_preop_scan.nii.gz")
        'patient123'
    """
    # Split the string using the underscore character as the delimiter
    parts = string.split("_")
    # Take the first part of the split string
    prefix = parts[0]
    return prefix


def initial_alignment(fixed_image, moving_image):
    """
    Initialize the alignment of two 3D images using a transformation based on their geometrical centers.

    This function sets up an initial alignment for two 3D medical images (a fixed image and a moving image) by aligning their centers. It uses the SimpleITK library to cast the moving image to the same pixel type as the fixed image, and then applies a 3D Euler transformation centered on the geometrical center of the images.

    Parameters:
    fixed_image (SimpleITK.Image): The image which serves as the reference frame.
    moving_image (SimpleITK.Image): The image to be aligned to the fixed image.

    Returns:
    SimpleITK.Transform: A 3D Euler transformation object representing the initial alignment based on the geometric centers of the images.

    Note:
    This function assumes that both input images are already in a spatial 3D context and are compatible for pixel casting and transformation operations.
    """
    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_image, moving_image.GetPixelID()),
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    return initial_transform


def apply_registration_mattesmutal(initial_transform, fixed_image, moving_image):
    """
    Perform image registration using the Mattes Mutual Information metric and a gradient descent optimizer.
    Parameters:
    initial_transform (SimpleITK.Transform): The initial transformation approximation between the fixed and moving images.
    fixed_image (SimpleITK.Image): The image which serves as the reference frame.
    moving_image (SimpleITK.Image): The image to be aligned to the fixed image.

    Returns:
    SimpleITK.Transform: The final transformation that best aligns the moving image to the fixed image according to the optimization process.

    """

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=1000
    )  # , estimateLearningRate=registration_method.EachIteration)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    final_transform = sitk.Euler3DTransform(initial_transform)
    registration_method.SetInitialTransform(final_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    )
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    return final_transform


def apply_registration_correlation(initial_transform, fixed_image, moving_image):
    """
    Perform image registration using the Cross Correlation metric and a gradient descent optimizer.

    Parameters:
    initial_transform (SimpleITK.Transform): The initial transformation approximation between the fixed and moving images.
    fixed_image (SimpleITK.Image): The image which serves as the reference frame.
    moving_image (SimpleITK.Image): The image to be aligned to the fixed image.

    Returns:
    SimpleITK.Transform: The final transformation that best aligns the moving image to the fixed image according to the optimization process.
    """

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1, numberOfIterations=1000
    )  # , estimateLearningRate=registration_method.EachIteration)
    # registration_method.SetOptimizerScalesFromPhysicalShift()

    final_transform = sitk.Euler3DTransform(initial_transform)
    registration_method.SetInitialTransform(final_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    )
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    return final_transform
