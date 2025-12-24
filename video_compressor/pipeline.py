import os
import csv
import cv2
import torch
import shutil
import subprocess
import numpy as np
import lpips

from typing import List, Tuple, Optional, Callable
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from scipy.stats import norm

from video_compressor.metrics import Metrics
from video_compressor.utils.progress import progress

class KeyframeSelector:
    def __init__(self, video_path: str, verbose: bool = True) -> None:
        self.video_path: str = video_path
        self.verbose: bool = verbose

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.lpips_model = lpips.LPIPS(net="alex").to(self.device)

        self.frame_pairs: int = 0
        self.metrics: Optional[np.ndarray] = None
        self.retained_indices: Optional[List[int]] = None
        self.reductions: List[dict] = []

        # Flags (kept intentionally)
        self.metrics_computed: bool = False
        self.metric_file_created: bool = False
        self.retained_indices_computed: bool = False
        self.retained_indices_file_created: bool = False
        self.output_video_created: bool = False

        # Paths
        self.metrics_file: Optional[str] = None
        self.output_video: Optional[str] = None

        # Directories
        self.metrics_dir = "metrics"
        self.plots_dir = "plots"
        self.temp_dir = "temp_keyframes"
        self.output_dir = "output_videos"

        self._ensure_dirs()

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _ensure_dirs(self) -> None:
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def _compute_thresholds(
        self,
        diffs: np.ndarray,
        deltas: np.ndarray,
        adapt_factor: float,
        abs_thres: Optional[float],
        delta_thres: Optional[float],
    ) -> Tuple[float, float]:

        if abs_thres is None:
            abs_thres = diffs.mean() + adapt_factor * diffs.std()

        if delta_thres is None:
            delta_thres = np.abs(deltas).mean() + adapt_factor * np.abs(deltas).std()

        return abs_thres, delta_thres

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def compute_metrics(self, callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_pairs = max(0, total_frames - 1)

        if self.frame_pairs == 0:
            raise ValueError("Video must contain at least two frames")

        self.metrics = np.zeros((self.frame_pairs, 4), dtype=np.float32)

        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")

        iterable = range(self.frame_pairs)
        if callback is None:
            iterable = progress(iterable, desc="Computing metrics", ncols=100)

        for i in iterable:
            ret, curr_frame = cap.read()
            if not ret:
                break

            stats = Metrics(prev_frame, curr_frame, self.device, self.lpips_model)
            self.metrics[i] = [
                stats.mse,
                stats.inv_ssim,
                stats.lpips,
                stats.difference,
            ]

            prev_frame = curr_frame
            
            if callback:
                callback((i + 1) / self.frame_pairs, f"Computing metrics: {i + 1}/{self.frame_pairs}")

        cap.release()
        self.metrics_computed = True
        return self.metrics

    def create_metric_file(self, output_path: Optional[str] = None) -> None:
        if not self.metrics_computed or self.metrics is None:
            raise RuntimeError("Metrics not computed")

        if output_path is None:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = os.path.join(self.metrics_dir, f"{video_name}_metrics.csv")

        self.metrics_file = output_path

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Frame_Pair", "MSE", "Inverse SSIM", "LPIPS", "Difference"]
            )
            for i in range(self.metrics.shape[0]):
                writer.writerow(
                    [
                        f"{i + 1}_vs_{i + 2}",
                        *[f"{v:.6f}" for v in self.metrics[i]],
                    ]
                )

        self.metric_file_created = True
        if self.verbose:
            print(f"Metrics file saved to {output_path}")

    # ------------------------------------------------------------------ #
    # Keyframe Selection
    # ------------------------------------------------------------------ #

    def select_keyframes(
        self,
        abs_thres: Optional[float] = None,
        delta_thres: Optional[float] = None,
        adapt_factor: float = 0.0,
        set_data: bool = True,
    ) -> Tuple[float, float, float]:

        if not self.metrics_computed or self.metrics is None:
            raise RuntimeError("Metrics not computed")

        diffs = self.metrics[:, 3]
        deltas = np.diff(diffs, prepend=diffs[0])

        abs_thres, delta_thres = self._compute_thresholds(
            diffs, deltas, adapt_factor, abs_thres, delta_thres
        )

        keep = [
            i
            for i in range(self.frame_pairs)
            if diffs[i] > abs_thres and abs(deltas[i]) > delta_thres
        ]

        keep = sorted(set([0, *keep, self.frame_pairs - 1]))

        if set_data:
            self.retained_indices = keep
            self.retained_indices_computed = True

        ratio = len(keep) / self.frame_pairs

        if self.verbose:
            print(
                f"Abs={abs_thres:.4f}, Δ={delta_thres:.4f} → "
                f"{len(keep)}/{self.frame_pairs} ({ratio * 100:.2f}%)"
            )

        return ratio, abs_thres, delta_thres

    def create_retained_indices_file(self, output_path: Optional[str] = None) -> None:
        if not self.retained_indices_computed or self.retained_indices is None:
            raise RuntimeError("Keyframes not selected")

        if output_path is None:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = os.path.join(
                self.metrics_dir, f"{video_name}_retained_indices.csv"
            )

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_Index"])
            for idx in self.retained_indices:
                writer.writerow([idx])

        self.retained_indices_file_created = True
        if self.verbose:
            print(f"Retained indices saved to {output_path}")

    # ------------------------------------------------------------------ #
    # Analysis & Visualization
    # ------------------------------------------------------------------ #

    def _plot_threshold_distribution(
        self, parameter: str, thresholds: List[float], retained_ratios: List[float]
    ) -> None:
        plt.figure(figsize=(12, 6))

        plt.plot(thresholds, retained_ratios, lw=2)
        plt.scatter(thresholds, retained_ratios, s=20)

        mu, sigma = np.mean(thresholds), np.std(thresholds)
        x = np.linspace(min(thresholds), max(thresholds), 300)
        y = norm.pdf(x, mu, sigma)
        y = y / y.max() * max(retained_ratios)

        plt.fill_between(x, y, alpha=0.2)

        plt.title(f"Frames Retained vs {parameter} Threshold")
        plt.xlabel(f"{parameter} Threshold")
        plt.ylabel("Frames Retained (%)")
        plt.grid(alpha=0.3)

        out = os.path.join(self.plots_dir, f"{parameter.lower()}_threshold_distribution.png")
        plt.savefig(out, dpi=300)
        plt.close()

    def set_reductions(self, n: int = 20) -> List[dict]:
        if not self.metrics_computed:
            raise RuntimeError("Metrics not computed")
            
        adapt_factors = np.linspace(-2.0, 5.0, n)
        self.reductions = []
        
        for f in adapt_factors:
            ratio, abs_t, delta_t = self.select_keyframes(
                adapt_factor=f, set_data=False
            )
            # Store reduction info (higher ratio = less reduction)
            # reduction % = (1 - ratio) * 100
            self.reductions.append({
                'reduction_percent': (1 - ratio) * 100,
                'ratio': ratio,
                'abs_thres': abs_t,
                'delta_thres': delta_t,
                'adapt_factor': f
            })
            
        return self.reductions

    def analyze_thresholds(self, num_factors: int = 20) -> None:
        adapt_factors = np.linspace(-2.0, 5.0, num_factors)
        retained_ratios, abs_vals, delta_vals = [], [], []

        for f in progress(adapt_factors, desc="Analyzing thresholds"):
            ratio, abs_t, delta_t = self.select_keyframes(
                adapt_factor=f, set_data=False
            )
            retained_ratios.append(ratio * 100)
            abs_vals.append(abs_t)
            delta_vals.append(delta_t)

        self._plot_threshold_distribution("Absolute", abs_vals, retained_ratios)
        self._plot_threshold_distribution("Delta", delta_vals, retained_ratios)

    def visualize_frames_fullscreen(
        self,
        start_frame: int = 0,
        num_frames: int = 36,
        skip: int = 0,
        cmap_name: str = "coolwarm",
    ) -> None:
        if self.metrics is None:
            raise RuntimeError("Metrics not computed")

        cap = cv2.VideoCapture(self.video_path)
        diffs = self.metrics[:, 3]
        norm_c = colors.Normalize(vmin=diffs.min(), vmax=diffs.max())
        mapper = cm.ScalarMappable(norm=norm_c, cmap=cm.get_cmap(cmap_name))

        frames, vals = [], []
        for i in range(num_frames):
            idx = start_frame + i * (skip + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            vals.append(diffs[min(idx - 1, len(diffs) - 1)])

        cap.release()

        cols = int(np.ceil(np.sqrt(len(frames))))
        rows = int(np.ceil(len(frames) / cols))

        plt.figure(figsize=(cols * 3.5, rows * 3.5))
        for i, (frame, v) in enumerate(zip(frames, vals)):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(frame)
            ax.axis("off")
            color = mapper.to_rgba(v)
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(5)
            ax.set_title(f"Δ={v:.3e}", fontsize=8)

        out = os.path.join(self.plots_dir, "frame_visualization.png")
        plt.savefig(out, dpi=300)
        plt.close()

    # ------------------------------------------------------------------ #
    # Video Creation
    # ------------------------------------------------------------------ #

    def _extract_frames(self, callback: Optional[Callable[[float, str], None]] = None) -> float:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ids = set(self.retained_indices or [])

        saved = 0
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_ids:
                cv2.imwrite(
                    os.path.join(self.temp_dir, f"frame_{saved:04d}.jpg"), frame
                )
                saved += 1
            
            if callback and i % 5 == 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                callback((i + 1) / total_frames, f"Extracting frames: {i + 1}/{total_frames}")

        cap.release()
        return fps

    def _encode_with_ffmpeg(self, output_path: str, fps: float) -> None:
        cmd =  [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", f"{self.temp_dir}/frame_%04d.jpg",
                "-c:v", "libx264",
                "-crf", "23",
                output_path,
            ]
        if self.verbose:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def get_sizes(self) -> None:
        if not self.output_video_created or self.output_video is None:
            raise RuntimeError("Output video not created")

        orig = os.path.getsize(self.video_path) / (1024 ** 2)
        new = os.path.getsize(self.output_video) / (1024 ** 2)

        return orig, new

    def create_compressed_video(self, callback: Optional[Callable[[float, str], None]] = None) -> None:
        if not self.retained_indices_computed:
            raise RuntimeError("Keyframes not selected")

        fps = self._extract_frames(callback=callback)
        name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.output_video = os.path.join(self.output_dir, f"{name}_keyframes.mp4")

        self._encode_with_ffmpeg(self.output_video, fps)
        shutil.rmtree(self.temp_dir)

        self.output_video_created = True

        if self.verbose:
            print(f"Keyframe video created at {self.output_video}")