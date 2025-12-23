import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scripts.select_keyframes import select_keyframes
from tqdm import tqdm
from scipy.stats import norm

def plot_threshold_distribution(parameter, threshold, retained_ratios):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12,6))

    plt.plot(threshold, retained_ratios, '-', color='blue', lw=2, label="% frames retained")
    plt.scatter(threshold, retained_ratios, s=15, color='blue', alpha=0.7)

    mu, sigma = np.mean(threshold), np.std(threshold)
    x_vals = np.linspace(min(threshold), max(threshold), 400)
    y_vals = norm.pdf(x_vals, mu, sigma)
    y_scaled = y_vals / max(y_vals) * max(retained_ratios)
    plt.fill_between(x_vals, y_scaled, alpha=0.12, color='orange', label="Threshold Distribution (scaled)")

    plt.title(f"Frames Retained vs {parameter} Threshold", fontsize=14, weight='bold')
    plt.xlabel(f"{parameter} Threshold", fontsize=12)
    plt.ylabel("Frames Retained (%)", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')

    plt.xticks(np.linspace(min(threshold), max(threshold), 12))
    plt.yticks(np.linspace(0, 100, 11))

    plt.legend(frameon=False)
    plt.tight_layout()
    plot_path = os.path.join("plots", f"{parameter.lower()}_threshold_distribution.png")
    plt.savefig(plot_path, dpi=300)

def analyze_thresholds(csv_path, verbose=True, save_all_keyframes=False):
    keep_indices, base_ratio, _, _ = select_keyframes(csv_path, adapt_factor=1.0)
    if(verbose):
        print(f"\nBase retention ratio: {base_ratio:.2f}%")

    adapt_factors = np.linspace(-2.0, 5.0, 20)
    
    retained_ratios = []
    abs_thresholds = []
    delta_thresholds = []
    for factor in tqdm(adapt_factors, desc="Testing absolute thresholds"):
        keep_indices, pct,(mean_a_diff, std_a_diff),(mean_d_diff, std_d_diff) = select_keyframes(csv_path, abs_thresh=None, delta_thresh=None, adapt_factor=factor, verbose=False)
        retained_ratios.append(pct)
        
        a_thres = mean_a_diff + factor * std_a_diff
        d_thres = mean_d_diff + factor * std_d_diff
        
        abs_thresholds.append(a_thres)
        delta_thresholds.append(d_thres)

        if(save_all_keyframes):
            os.makedirs("keyframes", exist_ok=True)

            save_keyframes_path = f"keyframe_{a_thres:.4f}_{d_thres:.4f}_{pct:.2f}.csv"
            final_path = os.path.join("keyframes", save_keyframes_path)
            
            ext = os.path.splitext(final_path)[1].lower()
            if ext == ".csv":
                with open(final_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Frame_Index"])
                    for idx in keep_indices:
                        writer.writerow([idx])
            else:
                np.savetxt(final_path, keep_indices, fmt="%d")
        
    plot_threshold_distribution("Absolute", abs_thresholds, retained_ratios)
    plot_threshold_distribution("Delta", delta_thresholds, retained_ratios)