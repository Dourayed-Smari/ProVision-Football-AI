# 🏆 ProVision-Football-AI

<div align="center">
  <h3>Developed by Dourayed Smari</h3>
  <img src="output_videos/demo.gif" width="800" />
  <p><strong>Advanced Computer Vision Pipeline for Professional Football Tactical Analysis</strong></p>
  <a href="output_videos/c1_out.mp4">📺 Watch Full Quality Video</a>
</div>

---

## 📖 Overview

**ProVision-Football-AI** is a state-of-the-art AI coaching tool designed to digitize the beautiful game. By transforming raw broadcast or tactical camera footage into structured, actionable telemetry, this system bridges the gap between video analysis and data science. 

Whether you are a scout analyzing player movements or a coach reviewing team shapes, ProVision automates the heavy lifting of manual video tagging.

## ✨ Key Features

### 👁️‍🗨️ Perception Engine (Vision)
*   **Multi-Object Detection:** Highly optimized YOLO models to detect Players, Goalkeepers, Referees, and the Match Ball.
*   **Robust Tracking:** Implementation of tracking algorithms (Kalman Filters & BoTSORT) to maintain player identities across frames, even during heavy occlusions.
*   **Pitch Keypoint Extraction:** Automatic detection of football field landmarks (penalty spots, lines, corners) to understand spatial depth.

### 🧠 Intelligence Engine (Tactics)
*   **Homography (2D Mapping):** Maps the 3D camera perspective onto a precise 2D tactical minimap. Calculates real-world coordinates in meters.
*   **Dynamic Team Classification:** Uses Unsupervised Machine Learning (K-Means Clustering) on player jersey colors to automatically assign teams dynamically without hardcoded rules.
*   **Advanced Analytics:** 
    *   Estimates player speeds (km/h) and distances covered.
    *   Possession calculation based on proximity algorithms.
    *   Automated tactical formation detection (e.g., 4-3-3 vs 3-5-2).

### 📊 Visualization & Output
*   **Cinematic Overlays:** Fully annotated video output with tracking IDs, team colors, and speed metrics.
*   **Tactical Artifacts:** Automatically generates:
    *   Player Heatmaps (Gaussian density maps)
    *   Passing Networks
    *   Comprehensive Match Stats Reports (HTML)

---

## ⚙️ Architecture

The project is built with modularity in mind, allowing independent upgrades to the vision or logic engines.

```text
ProVision-Football-AI/
├── analysis/               # Tactical algorithms (Formations, Heatmaps, Passes)
├── annotation/             # Rendering engine for video overlays
├── tracking/               # Object and Keypoint tracking logic
├── position_mappers/       # Homography and spatial math
├── models/                 # Neural Network weights (.pt files)
├── input_videos/           # Drop your raw footage here
└── output_videos/          # Rendered tactical videos and artifacts
```

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.10** (Recommended for stability with Deep Learning libraries)
*   CPU or NVIDIA GPU (The pipeline is optimized to run smoothly even on modern CPUs).

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/ProVision-Football-AI.git
    cd ProVision-Football-AI
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    *(For CPU execution)*
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics supervision opencv-python pandas scikit-learn
    ```

### Usage

1.  Place your raw match video in the `input_videos/` directory.
2.  Ensure your YOLO model weights are present in the `models/` directory.
3.  Execute the main engine:
    ```bash
    python main.py
    ```
4.  Find your fully analyzed video and tactical reports in the `output_videos/` folder.

---

## 🛠️ Built With

*   **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - Core detection engine
*   **[Supervision](https://github.com/roboflow/supervision)** - Video processing & CV utilities
*   **[PyTorch](https://pytorch.org/)** - Deep Learning framework
*   **[OpenCV](https://opencv.org/)** - Image manipulation

---

<div align="center">
  <i>Developed and maintained by <b>Dourayed Smari</b>.</i>
  <br>
  <p>Exploring the intersection of Computer Vision and Sports Science.</p>
</div>

---

## 👨‍💻 Developer Info
- **Project Lead:** Dourayed Smari
- **Role:** AI & Computer Vision Engineer
- **Focus:** Tactical Sports Analysis & Performance Digitization

---

