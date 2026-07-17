# Traffic Sign Recognition

A real-time German traffic sign classifier built with deep learning. The system takes a webcam feed and identifies traffic signs as they appear, using a convolutional neural network fine-tuned on the **German Traffic Sign Recognition Benchmark (GTSRB)** — 50,000+ images across 43 sign classes.

This project targets the computer-vision / perception engineering skills used in automotive AI (autonomous driving, driver-assistance systems).

## Status

Work in progress. Currently in **Phase 1 (Foundation)** of a 12-week roadmap.

- [x] Week 1 — Development environment & project setup
- [x] Week 2 — Machine learning fundamentals
- [x] Week 3 — Deep learning & CNNs (MNIST) — 97% test accuracy on handwritten digits
- [ ] Weeks 4-8 — Traffic Sign Recognition (this repo's main deliverable)

## Tech Stack

The project uses **Python 3.12** with **PyTorch** and **TorchVision** for deep learning, **OpenCV** for camera capture and image processing, **NumPy / Pandas / Matplotlib / Seaborn** for data handling and visualization, and **Jupyter** for exploration and prototyping.

## Project Structure

```
traffic-sign-recognition/
├── notebooks/        # Jupyter notebooks for exploration
├── src/              # Production Python scripts
├── models/           # Saved model checkpoints (gitignored)
├── data/             # Dataset (gitignored)
├── outputs/          # Figures, logs, results
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

## Datasets

The `data/` folder is gitignored, so datasets are not included in this repo. Here's how they work:

- **MNIST** (used in `notebooks/02_mnist_mlp.ipynb`) doesn't need a manual download. The notebook pulls it automatically into `data/` on first run through `torchvision.datasets.MNIST(download=True)`.
- **GTSRB** is the main dataset for this project — [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html), collected by the Institut für Neuroinformatik, Ruhr-Universität Bochum. It has 43 classes shot in real-world conditions (varying lighting, occlusion, rotation). Like MNIST, it downloads automatically into `data/` on first run through `torchvision.datasets.GTSRB(download=True)` (used in `notebooks/03_gtsrb_cnn.ipynb`).

## License

MIT
