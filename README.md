# 🎬 VidCompressorX

[![PyPI version](https://img.shields.io/pypi/v/vidcompressorx?style=for-the-badge)](https://badge.fury.io/py/vidcompressorx)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Intelligent video compression powered by perceptual metrics and adaptive keyframe selection.**

VidCompressorX is a Python library that uses computer vision and deep learning to intelligently compress videos by identifying and retaining only the most significant frames. By analyzing frame-to-frame differences using multiple perceptual metrics (MSE, SSIM, LPIPS), it achieves substantial compression ratios while maintaining visual quality.

## ✨ Features

- 🧠 **Multi-Metric Analysis** — Combines MSE, SSIM, and LPIPS for intelligent frame comparison
- 🎯 **Adaptive Thresholding** — Automatically determines optimal keyframe selection thresholds
- 📊 **Research-Friendly** — Extensive visualization and analysis tools for experimentation
- 🚀 **Production-Ready** — Clean API with proper state management and error handling
- 📓 **Notebook Compatible** — Progress bars automatically adapt to Jupyter environments
- ⚡ **GPU Accelerated** — CUDA support for faster LPIPS computation

## 🚀 Installation

```bash
pip install vidcompressorx
```

### Requirements

- Python 3.9 or higher
- FFmpeg (for video encoding)

**Install FFmpeg:**
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 📖 Quick Start

### Basic Usage

```python
from video_compressor import KeyframeSelector

# Initialize with your video
selector = KeyframeSelector('input_video.mp4')

# Compute frame-to-frame metrics
selector.compute_metrics()

# Select keyframes (adapt_factor controls aggressiveness)
# Higher values = more compression, lower values = more quality
selector.select_keyframes(adapt_factor=1.0)

# Create compressed video
selector.create_compressed_video()

# Check compression results
selector.get_sizes()
```

**Output:**
```
=== SIZE COMPARISON ===
Original: 45.23 MB
Keyframe: 8.91 MB
Reduction: 36.32 MB
Ratio: 5.08x
```

### Advanced Usage

```python
from video_compressor import KeyframeSelector

selector = KeyframeSelector('video.mp4', verbose=True)

# Step 1: Compute metrics
metrics = selector.compute_metrics()
selector.create_metric_file('output_metrics.csv')

# Step 2: Analyze threshold sensitivity
selector.analyze_thresholds(num_factors=20)  # Creates plots in plots/

# Step 3: Manual threshold control
selector.select_keyframes(
    abs_thres=50.0,      # Absolute difference threshold
    delta_thres=2.5,     # Rate-of-change threshold
    adapt_factor=None    # Disable adaptive thresholding
)

# Step 4: Export keyframe indices
selector.create_retained_indices_file('keyframes.csv')

# Step 5: Visualize frames
selector.visualize_frames_fullscreen(
    start_frame=0,
    num_frames=36,
    skip=10,
    cmap_name='viridis'
)

# Step 6: Create final video
selector.create_compressed_video()
```

## 🎯 How It Works

### 1. **Metric Computation**

VidCompressorX analyzes consecutive frame pairs using three complementary metrics:

- **MSE (Mean Squared Error)** — Pixel-level differences
- **Inverse SSIM** — Structural similarity changes  
- **LPIPS** — Perceptual similarity using deep learning (AlexNet)

These are combined into a weighted score:
```
difference = 0.5 × MSE + 0.3 × inv_SSIM + 0.2 × LPIPS
```

### 2. **Keyframe Selection**

Frames are selected based on two criteria:

- **Absolute Threshold** — Frame difference exceeds baseline
- **Delta Threshold** — Rate-of-change in difference is significant

```python
keep_frame if (difference > abs_threshold) AND (|Δdifference| > delta_threshold)
```

Adaptive thresholding automatically computes these based on video statistics:
```
threshold = mean(differences) + adapt_factor × std(differences)
```

### 3. **Video Reconstruction**

Selected keyframes are:
1. Extracted as JPEG images
2. Encoded with FFmpeg (H.264, CRF 23)
3. Compiled into final MP4 at original frame rate

## 📊 Understanding Adapt Factor

The `adapt_factor` parameter controls compression aggressiveness:

| Adapt Factor | Retention | Use Case |
|--------------|-----------|----------|
| `-2.0` to `0.0` | 80-95% | Minimal compression, high quality |
| `0.0` to `1.0` | 50-80% | Balanced compression |
| `1.0` to `3.0` | 20-50% | Aggressive compression |
| `3.0` to `5.0` | 5-20% | Maximum compression |

**Example:**
```python
# Conservative (high quality)
selector.select_keyframes(adapt_factor=0.5)  # ~70% frames retained

# Balanced
selector.select_keyframes(adapt_factor=1.5)  # ~40% frames retained

# Aggressive (high compression)
selector.select_keyframes(adapt_factor=3.0)  # ~15% frames retained
```

## 🔬 Experimentation Tools

VidCompressorX includes standalone scripts for research and analysis:

### Compute Metrics Only

```bash
python -m experiments.compute_metrics
```

```python
from experiments.compute_metrics import compute_video_metrics

compute_video_metrics(
    video_path='input.mp4',
    output_path='metrics.csv',
    verbose=True
)
```

### Analyze Threshold Sensitivity

```python
from experiments.keyframes_dist import analyze_thresholds

analyze_thresholds(
    csv_path='metrics.csv',
    save_all_keyframes=True  # Saves CSV for each threshold tested
)
```

Generates plots showing retention vs. threshold relationships.

### Batch Compression Analysis

```bash
python -m experiments.plot_compression
```

Processes multiple keyframe configurations and plots compression curves.

### Frame Visualization

```bash
python -m experiments.visualize_frames video.mp4 metrics.csv -n 40 -s 10 -c coolwarm
```

Creates a fullscreen grid of frames color-coded by motion intensity.

**Arguments:**
- `-k/--start`: Starting frame index (default: 500)
- `-n/--num`: Number of frames to display (default: 40)
- `-s/--skip`: Skip interval between frames (default: 0)
- `-c/--cmap`: Matplotlib colormap (default: 'coolwarm')

## 📁 Project Structure

```
vidcompressorx/
├── video_compressor/          # Core package
│   ├── __init__.py            # Public API exports
│   ├── pipeline.py            # KeyframeSelector class
│   ├── metrics.py             # Metrics computation
│   └── utils/
│       └── progress.py        # Environment-aware progress bars
├── experiments/               # Research tools
│   ├── compute_metrics.py     # Standalone metrics computation
│   ├── select_keyframes.py    # CLI keyframe selection
│   ├── keyframes_dist.py      # Threshold analysis
│   ├── plot_compression.py    # Batch compression analysis
│   ├── create_mp4.py          # Manual video creation
│   ├── visualize_frames.py    # Frame visualization
│   └── metrics_utils.py       # Shared metric utilities
├── pyproject.toml             # Package configuration
├── setup.py                   # Setup script
└── README.md                  # This file
```

## 🎓 API Reference

### `KeyframeSelector`

**Initialization:**
```python
selector = KeyframeSelector(video_path, verbose=True)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `compute_metrics()` | Analyzes all frame pairs | `np.ndarray` |
| `create_metric_file(output_path)` | Exports metrics to CSV | `None` |
| `select_keyframes(abs_thres, delta_thres, adapt_factor)` | Selects keyframes | `(ratio, abs_t, delta_t)` |
| `create_retained_indices_file(output_path)` | Exports indices to CSV | `None` |
| `analyze_thresholds(num_factors)` | Threshold sensitivity analysis | `None` |
| `visualize_frames_fullscreen(...)` | Frame grid visualization | `None` |
| `create_compressed_video()` | Generates final video | `None` |
| `get_sizes()` | Prints size comparison | `None` |

**State Flags:**
- `metrics_computed`: Metrics calculation complete
- `metric_file_created`: Metrics CSV exported
- `retained_indices_computed`: Keyframe selection complete
- `retained_indices_file_created`: Indices CSV exported
- `output_video_created`: Final video generated

### `Metrics`

**Initialization:**
```python
from video_compressor import Metrics

metrics = Metrics(frame1, frame2, device='cuda', lpips_model=model)
```

**Attributes:**
- `mse`: Mean Squared Error
- `inv_ssim`: Inverse SSIM
- `lpips`: LPIPS score
- `difference`: Combined weighted metric

## 🔧 Troubleshooting

### FFmpeg Not Found
```
Error: FFmpeg not installed
```
**Solution:** Install FFmpeg (see Installation section)

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Process shorter videos or use CPU:
```python
# Force CPU usage
import torch
torch.cuda.is_available = lambda: False
```

### Low Compression Ratio
**Solution:** Increase `adapt_factor`:
```python
selector.select_keyframes(adapt_factor=2.5)
```

### Too Much Compression
**Solution:** Decrease `adapt_factor` or set manual thresholds:
```python
selector.select_keyframes(
    abs_thres=30.0,
    delta_thres=1.5
)
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Satvik Virmani**

Feel free to reach out for questions, suggestions, or collaboration opportunities!

## 🙏 Acknowledgments

- **LPIPS** — Zhang et al. for the perceptual similarity metric
- **OpenCV** — For video processing capabilities
- **FFmpeg** — For video encoding

## 🌟 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Roadmap

- [ ] Support for more codecs (H.265, VP9, AV1)
- [ ] Real-time preview during selection
- [ ] Configurable metric weights
- [ ] Scene detection integration
- [ ] Multi-video batch processing
- [ ] Web interface for non-programmers

---

<div align="center">

**If you find VidCompressorX useful, please consider giving it a ⭐ on GitHub!**

</div>
