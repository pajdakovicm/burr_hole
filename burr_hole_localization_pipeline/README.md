# Trepanation site localization

This document describes the pipeline for trepanation site localization using pairs of preoperative and postoperative CT scans. The pipeline consists of the following steps:
1. Window clipping of both preoperative and postoperative CT scans.
2. Skull preprocessing.
3. Image registration. 
4. Dataset resampling.
5. Application of Anisotropic Diffusion filter.
6. Template matching.
7. Region properties analysis. 

## Installation

Python 3.10 was used.
To install all necessary libraries, run:

```bash
pip install -r requirements.txt
```

## Usage

To perform window clipping, run:
```bash
cd burr_hole_localization_pipeline/window_clipping

python3 clip.py [--help] --input dir INPUT_DIR_PATH --output_dir OUTPUT_DIR _PATH   
--min_clip -MIN_HU_VALUE --max_clip MAX_HU_VALUE
```
To see all available command-line options and their descriptions, use the --help flag.  
To perform image registration, run:
```bash
cd burr_hole_localization_pipeline/registration
```
To run the registration, you need to run a Jupyter Notebook file apply registration.ipynb.
The notebook contains detailed instructions for each step of the process. You
can either register images individually (useful for testing) or process the entire
dataset. This implementation is the [Introduction to Registration - SimpleITK](https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html)
 methodology, where you can find more detailed explanations. 
Input data for registration are output data of the previous step - clipped CT images.  
Registration will create two different directories: images (contains registered images) and transformations (contains transformations used for the registration).

The skull preprocessing step involves removal of hematoma by thresholding (for both registered and preoperative images), subtraction of registered and preoperative image wihtout  hematoma (outputs of previous step) and cut of lower parts of the skull. To remove hematoma, run:
```bash
cd burr_hole_localization_pipeline/skull_preprocessing
python3 hematoma_removal.py [--help] --input_dir INPUT_DIR_PATH 
--output_dir OUTPUT_DIR_PATH --threshold HU_THRESHOLD_VALUE
```
To subtract the pairs of images, run:
```bash
cd burr_hole_localization_pipeline/skull_preprocessing
python3 subtraction.py [--help] --input_dir_registered INPUT_REG_DIR_PATH 
--input_dir_preop INPUT_PREOP_DIR_PATH 
--output_dir OUTPUT_DIR_SUBTRACTED_PATH 
```
To cut the lower parts of the skull, run:
```bash
cd burr_hole_localization_pipeline/skull_preprocessing
python3 cut_skull.py [--help] --input_dir INPUT_DIR_PATH 
--output_dir OUTPUT_DIR_PATH --cutoff CUTTOFF_VALUE
```
Skull cutting is applied to subtracted images.

To resample the dataset, run:
```bash
cd burr_hole_localization_pipeline/resample
python3 resample_ds.py [--help] --input_dir INPUT_DIR_PATH 
--output_dir OUTPUT_DIR_PATH --spacing x_axis y_axis z_axis  
 --interpolator INTERPOLATOR_TYPE
```
Resampling is applied after skull cutting, for efficiency.

To apply Anisotropic Diffusion filter, run:
```bash
cd burr_hole_localization_pipeline/blob_extraction
python3 filtering.py [--help] --input_dir INPUT_DIR_PATH —output_dir OUTPUT_DIR_PATH

```
To extract the blob with the template matching technique, run:
```bash
cd burr_hole_localization_pipeline/blob_extraction
python3 template_matching.py [--help] --input_dir INPUT_DIR_PATH
 —output_dir OUTPUT_DIR_PATH —-trehshold THRESHOLD_VALUE --template TEMPLATE_PATH
 ```
To exclude false positives, apply region properties and run:
```bash
cd burr_hole_localization_pipeline/blob_extraction
python3 region_props.py --input_dir INPUT_DIR_PATH   
output_dir OUTPUT_DIR_PATH --area_min MIN_THRESHOLD_AREA --area_max  MAX_THRESHOLD_AREA 
 --csv_filename CSV_FILE_PATH --aspect_ratio_max_min ASPECT_RATIO_MAX_MIN_THRESHOLD  
 --aspect_ratio_mid_min ASPECT_RATIO_MID_MIN_THRESHOLD --solidity_threshold SOLIDITY_THRESHOLD
```
The final output directory contains the automatically obtained labels.

## Evaluation
For quantitative evaluation of the results, run:
```bash
cd burr_hole_localization_pipeline/labels_test 
python3 distance_error.py [—help] —gt_dir GT_DIR_PATH 
 —pred_dir PRED_DIR_PATH —output_csv CSV_FILE_PATH
```
This directory contains *gt_labels_analysis.py* script which served mainly for visualisations and analysis of ground truth labels. 
Besides, script *gt_to_preop_space.py* applies registration to ground truth labels, since they are in the space of postoperative images. Resampling of GT labels is applied after this step. 

To statistically compare if there is a siginificant difference between errors obtained from GT dataset 1 and GT dataset 2. In file *statistical analysis.py* you can set the paths and perform evaluation.  
