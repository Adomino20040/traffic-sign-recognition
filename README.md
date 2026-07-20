# Traffic Sign Recognition

A real-time German traffic sign classifier built with deep learning. The system takes a webcam feed and identifies traffic signs as they appear, using a convolutional neural network fine-tuned on the **German Traffic Sign Recognition Benchmark (GTSRB)**, a dataset of 50,000+ images across 43 sign classes.

This project targets the computer-vision / perception engineering skills used in automotive AI (autonomous driving, driver-assistance systems).

## Status

- [x] Week 1: Development environment and project setup
- [x] Week 2: Machine learning fundamentals
- [x] Week 3: Deep learning fundamentals. MNIST with a fully-connected net, 97% test accuracy on handwritten digits
- [x] Week 4: First CNN on GTSRB, **91.15% test accuracy** (vs 72.4% MLP baseline) with data augmentation and best-epoch checkpointing
- [x] Weeks 5-6: Production scripts (`model.py`, `predict.py`, `webcam_demo.py`) and real-time webcam demo with snapshot saving
- [ ] Weeks 7-8: Accuracy improvements (transfer learning, hyperparameter tuning)

## Demo

<table>
  <tr>
    <td><img src="docs/stop_sign_demo.png" width="350"></td>
    <td><img src="docs/turn_left_demo.png" width="350"></td>
  </tr>
  <tr>
    <td align="center">Stop sign at 99.97% confidence</td>
    <td align="center">Turn left ahead sign at 93.32% confidence</td>
  </tr>
</table>

## Tech Stack

The project uses **Python 3.12** with **PyTorch** and **TorchVision** for deep learning, **OpenCV** for camera capture and image processing, **NumPy / Pandas / Matplotlib / Seaborn** for data handling and visualization, and **Jupyter** for exploration and prototyping.

## Project Structure

```
traffic-sign-recognition/
├── notebooks/        # Jupyter notebooks for exploration
│   ├── 01_first_classifier.ipynb
│   ├── 02_mnist_mlp.ipynb
│   └── 03_gtsrb_cnn.ipynb
├── src/              # Production Python scripts
│   ├── model.py          # ConvNet architecture + model loader
│   ├── predict.py        # Single-image classifier
│   ├── webcam_demo.py    # Real-time webcam classifier
│   └── gtsrb_classes.py  # 43 class names
├── docs/             # Demo screenshots and assets
├── models/           # Saved model checkpoints (gitignored)
├── data/             # Dataset (gitignored)
├── outputs/          # Your own demo captures (gitignored)
├── requirements.txt  # Dependencies
├── test_setup.py     # Environment verification script
└── README.md
```

## Setup

Start by cloning the repo:

```bash
git clone <your-repo-url>
cd traffic-sign-recognition
```

Create a virtual environment with Python 3.12 and activate it:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then verify everything works:

```bash
python test_setup.py
```

## Usage

**Classify a single image:**
```bash
python src/predict.py path/to/sign.jpg
```

**Real-time webcam demo:**
```bash
python src/webcam_demo.py
```
Press **S** to save a snapshot, **ESC** to quit.

## Datasets

The `data/` folder is gitignored, so datasets are not included in this repo. Here's how they work:

- **MNIST** (used in `notebooks/02_mnist_mlp.ipynb`) doesn't need a manual download. The notebook pulls it automatically into `data/` on first run through `torchvision.datasets.MNIST(download=True)`.
- **GTSRB** is the main dataset for this project: the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html), collected by the Institut für Neuroinformatik, Ruhr-Universität Bochum. It has 43 classes shot in real-world conditions (varying lighting, occlusion, rotation). Like MNIST, it downloads automatically into `data/` on first run through `torchvision.datasets.GTSRB(download=True)` (used in `notebooks/03_gtsrb_cnn.ipynb`).

## License

MIT
