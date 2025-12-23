import cv2
import csv
import os
import torch
from tqdm import tqdm
from scripts.metrics_utils import compute_mse, compute_inv_ssim, compute_lpips, load_lpips_model

def compute_video_metrics(video_path, output_path, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(verbose):
        print(f"INFO: Using device: {device} for LPIPS computation.")

    loss_fn_alex = load_lpips_model(device)
    if(verbose):
        print("INFO: LPIPS model loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_pairs = total_frames - 1 if total_frames > 0 else None

    try:
        with open(output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ["Frame_Pair", "MSE", "Inverse SSIM", "LPIPS"]
            csv_writer.writerow(header)
            if(verbose):
                print(f"INFO: Output will be saved to '{output_path}'")

            ret, prev_frame = cap.read()
            if not ret:
                print("Error: Could not read the first frame.")
                cap.release()
                return

            frame_count = 1

            with tqdm(total=total_pairs, desc="Processing Frame Pairs", ncols=100) as pbar:
                while True:
                    ret, curr_frame = cap.read()
                    if not ret:
                        break

                    mse_val = compute_mse(prev_frame, curr_frame)
                    inv_ssim_val = compute_inv_ssim(prev_frame, curr_frame)
                    lpips_val = compute_lpips(prev_frame, curr_frame, loss_fn_alex, device)

                    frame_pair_label = f"{frame_count}_vs_{frame_count + 1}"
                    csv_writer.writerow([
                        frame_pair_label,
                        f"{mse_val:.6f}",
                        f"{inv_ssim_val:.6f}",
                        f"{lpips_val:.6f}"
                    ])

                    prev_frame = curr_frame
                    frame_count += 1
                    pbar.update(1)

    except IOError as e:
        print(f"Error writing CSV: {e}")
    finally:
        cap.release()
        if(verbose):
            print(f"\nVideo processing complete. Results saved to '{os.path.abspath(output_path)}'.")