import cv2
import csv
import os
import shutil
import subprocess
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

KEYFRAME_DIR = "./keyframes/"
INPUT_VIDEO = "../video/videos/videoplayback.mp4"
TEMP_DIR = "temp_keyframes"


def extract_retention(filename):
    """Extract last float number from filename: keyframe_A_B_retention.csv"""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", filename)
    return float(nums[-1])


def read_keyframes(csv_path):
    frames = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                frames.append(int(row[0]))
    return sorted(list(set(frames)))


def extract_frames(input_path, frame_indices, temp_dir):
    """Extract keyframes and save as JPGs."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise Exception("Could not open input video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_set = set(frame_indices)

    saved_count = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_id_set:
            out_path = os.path.join(temp_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1

    cap.release()
    return fps, saved_count


def encode_with_ffmpeg(output_path, fps, temp_dir):
    """Encode frames using H.265 for maximum compression."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{temp_dir}/frame_%04d.jpg",
        "-c:v", "libx265",
        "-preset", "medium",
        "-crf", "28",
        "-tag:v", "hvc1",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def batch_process(keyframe_dir, input_video, temp_dir):
    results = []

    csv_files = [f for f in os.listdir(keyframe_dir) if f.endswith(".csv")]

    print(f"\nFound {len(csv_files)} keyframe CSV files.")

    for csv_file in tqdm(csv_files, desc="Processing keyframes"):
        csv_path = os.path.join(keyframe_dir, csv_file)
        retention = extract_retention(csv_file)

        keyframes = read_keyframes(csv_path)

        # temp output filename
        output_video = os.path.join(
            keyframe_dir, csv_file.replace(".csv", ".mp4")
        )

        fps, saved = extract_frames(input_video, keyframes, temp_dir)
        encode_with_ffmpeg(output_video, fps=fps, temp_dir=temp_dir)

        # compute sizes
        csv_size = size_mb(csv_path)
        mp4_size = size_mb(output_video)
        total_size = csv_size + mp4_size

        original_size = size_mb(input_video)
        compression_ratio = original_size / total_size if total_size > 0 else 0

        results.append([retention, total_size, compression_ratio])

        shutil.rmtree(temp_dir)

    # create dataframe
    df = pd.DataFrame(results, columns=["Retention (%)", "Total Size (MB)", "Compression Ratio"])
    df.sort_values("Retention (%)", inplace=True)

    return df


def plot_results(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["Retention (%)"], df["Total Size (MB)"], marker="o")
    plt.xlabel("Frame Retention (%)")
    plt.ylabel("Total Compressed Size (MB)")
    plt.title("Compression Size vs Frame Retention")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/compression_size_vs_retention.png", dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df["Retention (%)"], df["Compression Ratio"], marker="o")
    plt.xlabel("Frame Retention (%)")
    plt.ylabel("Total Compressed Size (MB)")
    plt.title("Compression Size vs Frame Retention")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/compression_ratio_vs_retention.png", dpi=300)

    print("\n=== RESULTS DATAFRAME ===")
    print(df)


if __name__ == "__main__":
    df = batch_process("./keyframes", "../video/videos/videoplayback.mp4", "temp_keyframes")
    plot_results(df)