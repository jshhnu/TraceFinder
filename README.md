# 🔬 TraceFinder: AI-Based Forensic Scanner Identification System

TraceFinder is an advanced, digital document forensics application designed to identify the exact source hardware scanner model used to generate a digital document scan. By treating individual scanner sensors as biometric identities, the system extracts hardware-intrinsic anomalies to validate document legitimacy with deep precision.

---

## System Architecture Overview

The platform is engineered across a detached, hardware-accelerated full-stack architecture, bridging classical image signal processing with deep learning classification models.

###  Core Component Layout
* **Frontend Dashboard:** A responsive, component-driven Single Page Application (SPA) built using **React.js** to handle asynchronous multipart form-data uploads and display dynamic inference metrics.
* **Backend API Gateway:** A high-concurrency **Flask API** wrapper executing loopback routing pipelines, serving prediction arrays securely over standardized JSON communication frames.
* **Deep Forensic Model:** A custom-adapted 50-layer Residual Network (**ResNet-50**) backbone trained to isolate hardware-intrinsic biometric fingerprints across major scanner manufacturers (e.g., Canon, HP).

---

## 🛠️ Deep Technical Engineering Highlights

### 1. Forensic Signal Processing & Noise Isolation
To prevent the deep learning model from learning semantic text details or standard page content, the pipeline implements an **Adaptive $5 \times 5$ Spatial Wiener Deconvolution Filter**. This filter strips away high-frequency text boundaries and structural graphics, cleanly isolating the underlying residual high-frequency sensor noise anomalies—known as Photo-Response Non-Uniformity (PRNU).

### 2. Neural Network Input Layer Adaptation
The standard ResNet-50 backbone is natively designed for 3-channel RGB imagery. Because forensic noise residuals are highly localized structural signals, the model's primary convolution block was explicitly rewritten to accept a **single-channel grayscale luminance matrix**, optimizing memory usage and increasing structural feature extraction performance.

### 3. Dedicated Hardware Infiltration & Performance
* **Inference Pipeline:** Configured with strict internal thread constraints to safely bypass OpenBLAS memory allocation blocks.
* **Hardware Acceleration:** Fully integrated with PyTorch's **CUDA API wrappers**, routing tensor matrices directly onto a local **NVIDIA RTX 3050 Laptop GPU** to achieve sub-second processing latencies.

---

##  Tech Stack & Frameworks

* **Frontend:** React.js, HTML5, CSS3, JavaScript (ES6)
* **Backend API:** Python, Flask, Flask-CORS
* **Deep Learning Engine:** PyTorch, Torchvision, CUDA Core Execution Runtime
* **Computer Vision Processing:** OpenCV (cv2), NumPy, SciPy (Signal Processing Submodules)
* **Development IDE:** PyCharm Professional

---

##  Getting Started & Local Installation

### Prerequisites
* Python 3.10+
* Node.js & npm
* NVIDIA CUDA Toolkit (Highly recommended for hardware-accelerated processing)

### 1. Backend Server Setup
```bash
# Clone the repository
git clone [https://github.com/jshhnu/TraceFinder.git](https://github.com/jshhnu/TraceFinder.git)
cd TraceFinder

# Set up virtual environment
python -m venv .venv
source .venv/scripts/activate  # On Windows use: .venv\Scripts\activate

# Install required modules
pip install torch torchvision opencv-python flask flask-cors numpy scipy

# Start the Flask API
python app.py
