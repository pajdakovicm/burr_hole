from scipy.stats import shapiro
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

## evaluate if there is a significant difference between distance errors of GT1 vs. auomatatically obtained
## labels and GT2 vs. automatically obtained labels


def statistical_evaluation(gt_1_file_path, gt_2_file_path):
    gt1_df = pd.read_csv(gt_1_file_path)
    gt2_df = pd.read_csv(gt_2_file_path)

    # merge on common column
    merged = pd.merge(gt1_df, gt2_df, on="Filename", suffixes=("_gt1", "_gt2"))

    # compute difference between errors
    merged["diff"] = (
        merged["Euclidean Distance Error (mm)_gt1"]
        - merged["Euclidean Distance Error (mm)_gt2"]
    )

    # check normality
    stat, p_normal = shapiro(merged["diff"])
    print(f"Shapiro-Wilk p = {p_normal:.4f}, W={stat:.4f}")

    # choose appropriate test
    if p_normal > 0.05:
        t_stat, p_t = ttest_rel(
            merged["Euclidean Distance Error (mm)_gt1"],
            merged["Euclidean Distance Error (mm)_gt2"],
        )
        print(f"Paired t-test: t = {t_stat:.3f}, p = {p_t:.4f}")
    else:
        w_stat, p_w = wilcoxon(
            merged["Euclidean Distance Error (mm)_gt1"],
            merged["Euclidean Distance Error (mm)_gt2"],
        )
        print(f"Wilcoxon test: stat = {w_stat:.3f}, p = {p_w:.4f}")


if __name__ == "__main__":
    # file paths to ground truth labels 1 and 2
    gt_1 = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/csv_files/distance_error_0.8.csv"
    gt_2 = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/csv_files/distance_error_0.8_gt2.csv"
    statistical_evaluation(gt_1_file_path=gt_1, gt_2_file_path=gt_2)
