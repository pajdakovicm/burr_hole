from scipy.stats import shapiro
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu

## evaluate if there is a significant difference between distance errors of GT1 vs. auomatatically obtained
## labels and GT2 vs. automatically obtained labels


def statistical_evaluation1(gt_1_file_path, gt_2_file_path, paired=True):
    gt1_df = pd.read_csv(gt_1_file_path)
    gt2_df = pd.read_csv(gt_2_file_path)

    # Merge based on 'Filename' to ensure alignment
    merged = pd.merge(gt1_df, gt2_df, on="Filename", suffixes=("_gt1", "_gt2"))

    col1 = "Euclidean Distance Error (mm)_gt1"
    col2 = "Euclidean Distance Error (mm)_gt2"

    if paired:
        # Compute difference for normality check
        merged["diff"] = merged[col1] - merged[col2]

        # Shapiro-Wilk test for normality
        stat, p_normal = shapiro(merged["diff"])
        print(f"[Normality Test] Shapiro-Wilk p = {p_normal:.4f}, W = {stat:.4f}")

        # Paired test: parametric or non-parametric
        if p_normal > 0.05:
            t_stat, p_t = ttest_rel(merged[col1], merged[col2])
            print(f"[Paired t-test] t = {t_stat:.3f}, p = {p_t:.4f}")
        else:
            w_stat, p_w = wilcoxon(merged[col1], merged[col2])
            print(f"[Wilcoxon test] stat = {w_stat:.3f}, p = {p_w:.4f}")
    else:
        # Independent test: check both groups separately for normality
        stat1, p1 = shapiro(merged[col1])
        stat2, p2 = shapiro(merged[col2])
        print(f"[Normality Test - gt1] p = {p1:.4f}, W = {stat1:.4f}")
        print(f"[Normality Test - gt2] p = {p2:.4f}, W = {stat2:.4f}")

        if p1 > 0.05 and p2 > 0.05:
            t_stat, p_t = ttest_ind(merged[col1], merged[col2])
            print(f"[Independent t-test] t = {t_stat:.3f}, p = {p_t:.4f}")
        else:
            u_stat, p_u = mannwhitneyu(
                merged[col1], merged[col2], alternative="two-sided"
            )
            print(f"[Mann-Whitney U test] U = {u_stat:.3f}, p = {p_u:.4f}")


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
    gt_2 = "/Users/marijapajdakovic/Desktop/burr_hole_project/results/csv_files/gt1_vs_gt2.csv"
    statistical_evaluation1(gt_1_file_path=gt_1, gt_2_file_path=gt_2)
