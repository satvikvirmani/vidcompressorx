import cv2
import csv
import os
import shutil
import subprocess

KEYFRAME_CSV = "./keyframes/keyframe_58.1495_1.7910_52.67.csv"
INPUT_VIDEO = "../video/videos/videoplayback.mp4"
OUTPUT_VIDEO = "../video/videos/keyframes.mp4"
TEMP_DIR = "temp_keyframes"

def read_keyframes(csv_path):
    frames = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)    # skip header
        for row in reader:
            if row:
                frames.append(int(row[0]))
    return sorted(list(set(frames)))


def extract_frames(input_path, frame_indices):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        raise Exception("Could not open input video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_set = set(frame_indices)

    print(f"Extracting {len(frame_indices)} keyframes...")

    saved_count = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_id_set:
            out_path = os.path.join(TEMP_DIR, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1

    cap.release()
    print(f"Saved {saved_count} frames.")
    return fps


def encode_with_ffmpeg(output_path, fps=30):
    print("Encoding with FFmpeg (H.264)...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{TEMP_DIR}/frame_%04d.jpg",
        "-c:v", "libx265",
        "-preset", "medium",
        "-crf", "28",
        "-tag:v", "hvc1",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("FFmpeg video created.")


def compare_sizes(original, filtered):
    def size_mb(path):
        return os.path.getsize(path) / (1024 * 1024)

    orig = size_mb(original)
    filt = size_mb(filtered)

    print("\n=== FILE SIZE COMPARISON ===")
    print(f"Original video:        {orig:.2f} MB")
    print(f"Keyframe video:        {filt:.2f} MB")
    print(f"Size reduction:        {orig - filt:.2f} MB")
    print(f"Compression ratio:     {orig / filt:.2f}x")


if __name__ == "__main__":
    keyframes = read_keyframes(KEYFRAME_CSV)
    print("Loaded keyframes:", keyframes[:20], "..." if len(keyframes) > 20 else "")

    fps = extract_frames(INPUT_VIDEO, keyframes)
    encode_with_ffmpeg(OUTPUT_VIDEO, fps=fps)

    compare_sizes(INPUT_VIDEO, OUTPUT_VIDEO)

    shutil.rmtree(TEMP_DIR)

    print(f"\nSaved compressed video: {OUTPUT_VIDEO}")