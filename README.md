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

- **Python 3.12**
- **PyTorch** + **TorchVision** — deep learning & pretrained models
- **OpenCV** — camera capture and image processing
- **NumPy / Pandas / Matplotlib / Seaborn** — data handling & visualization
- **Jupyter** — exploration and prototyping

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

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd traffic-sign-recognition
   ```
2. Create and activate a virtual environment (Python 3.12):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify the setup:
   ```bash
   python test_setup.py
   ```

## Dataset

[GTSRB — German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html), collected by the Institut für Neuroinformatik, Ruhr-Universität Bochum. 43 classes, real-world conditions (varying lighting, occlusion, rotation).

## License

MIT
