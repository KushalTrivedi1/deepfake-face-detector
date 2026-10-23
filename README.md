# Deepfake Face Detector using MobileNetV2

**B.Tech Idea Lab Project — Ramdeobaba University**

Lightweight real-vs-fake **face** image classifier using transfer learning with **MobileNetV2**, compared against a baseline CNN trained from scratch.

| Item | Detail |
|------|--------|
| Task | Binary classification: Real vs AI-generated face |
| Main model | MobileNetV2 (ImageNet pretrained, two-stage fine-tune) |
| Baseline | Small CNN from scratch |
| Face test accuracy | **~90.89%** on 20,000 held-out face images |
| Training data | Mixed face set (~47k images from 140K Real/Fake Faces + extra faces) |
| Demo | Gradio web UI |
| Framework | TensorFlow / Keras |

> Academic proof-of-concept. Not a production forensic tool.

---

## Team

| Name | Roll No. | Contribution |
|------|----------|--------------|
| Kushal Trivedi | B2-52 | Model training, evaluation, report coordination |
| Koustubh Bhisade | B2-49 | Data pipeline, preprocessing, experiments |
| Pankaj Agrawal | B2-55 | Baseline CNN, comparison metrics |
| Varad Kawtikwar | B2-63 | Grad-CAM, analysis, documentation |
| Harsh Gaikwad | B3-43 | Gradio UI, demo, testing support |

**Guide:** Prof. Tejaswini Akulwar  
**Department:** Data Science and Engineering, Ramdeobaba University, Nagpur

---

## Repository contents

```
deepfake-face-detector/
├── README.md
├── requirements.txt
├── .gitignore
└── IDEA_LAB_PROJECT.ipynb   # Full Colab notebook (Sections 1–7 + Gradio)
```

### Notebook sections

1. **Setup** — mixed precision, Drive mount, Kaggle download (140K faces)
2. **Baseline CNN** — train from scratch, evaluate, save
3. **MobileNetV2** — transfer learning (frozen → fine-tune) on mixed faces
4. **Model comparison** — accuracy, size, inference time
5. **Evaluation** — classification report, ROC, confusion matrix, confidence
6. **Interpretability** — Grad-CAM
7. **Demo** — Gradio UI for upload & predict

---

## Datasets

| Dataset | Role |
|---------|------|
| [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Primary train / valid / test |
| Extra face corpus (optional) | Diversity in mixed training |

You need a Kaggle account and `kaggle.json` API token to download data inside Colab.

---

## How to run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `IDEA_LAB_PROJECT.ipynb` **or** open from GitHub:
   - File → Open notebook → GitHub → paste this repo URL.
3. Runtime → Change runtime type → **GPU** (T4).
4. Run cells in order:
   - Mount Drive
   - Upload `kaggle.json` when prompted
   - Train / evaluate (or load saved models from Drive if already trained)
5. Last cell launches **Gradio** with a public share link.

### Saved models (on your Drive after training)

- `mobilenetv2_internet_faces.h5` — final face detector (~90.9% test acc)
- `baseline_cnn.h5` — baseline CNN

---

## Local setup (optional)

```bash
git clone https://github.com/<YOUR_USERNAME>/deepfake-face-detector.git
cd deepfake-face-detector
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook IDEA_LAB_PROJECT.ipynb
```

GPU strongly recommended for training. CPU inference for Gradio is possible after training.

---

## Important notes

- **Preprocessing:** MobileNetV2 must use `mobilenet_v2.preprocess_input` (range ≈ [-1, 1]). Do **not** only divide by 255 for that model.
- **Augmentation** should be applied on raw images **before** MobileNet preprocessing.
- Results are strongest on **face** images similar to training data. Arbitrary non-face or unseen-generator images may be wrong.
- This is a research / academic demo, not legal evidence.

---

## Results (summary)

| Model | Face test accuracy |
|-------|--------------------|
| Baseline CNN | Lower (from-scratch reference) |
| MobileNetV2 (fine-tuned) | **~90.89%** |

---

## License

Academic / educational use for Ramdeobaba University Idea Lab project.  
Datasets remain under their original Kaggle licenses.

---

## Citation

If you use this work, please cite the project report / conference draft:

```
Trivedi, K., Bhisade, K., Agrawal, P., Kawtikwar, V., Gaikwad, H.
(2026). Deepfake Face Detector using MobileNetV2 Transfer Learning.
B.Tech Idea Lab Project, Ramdeobaba University, Nagpur.
Guide: Prof. Tejaswini Akulwar.
```
