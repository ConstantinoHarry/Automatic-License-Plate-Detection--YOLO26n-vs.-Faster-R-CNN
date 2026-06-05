# 🚗 Automatic License Plate Detection: YOLO26n vs. Faster R-CNN

This project systematically benchmarks two deep learning object detection architectures—a lightweight single-stage detector (YOLO26n) and a canonical two-stage detector (Faster R-CNN with a ResNet-50 FPN backbone)—for the task of Automatic License Plate Recognition (ALPR)[cite: 1]. 

The evaluation spans static detection quality, computational throughput on GPU hardware, and end-to-end recognition accuracy through an integrated Optical Character Recognition (OCR) pipeline using EasyOCR[cite: 1].

## 📊 Project Overview & Key Findings

The central challenge in ALPR is finding a detector that is tight enough to support downstream OCR and fast enough for real-time video processing[cite: 1]. Evaluated on an identical 87-image held-out validation split[cite: 1], the project reveals distinct tradeoffs:

*   **Accuracy vs. Speed:** Faster R-CNN achieves higher aggregate localization quality (mean IoU 0.7233 vs. 0.6679) and a significantly lower catastrophic-failure rate (6.9% vs. 17.2%)[cite: 1]. YOLO26n, however, delivers a 13.05x per-frame inference speed-up (11.1 ms vs. 144.9 ms on an NVIDIA T4)[cite: 1].
*   **The OCR Disconnect:** A critical finding is that geometric metrics (like IoU) do not fully determine end-to-end OCR outcomes[cite: 1]. In generalization testing, YOLO26n produced tighter bounding box crops that yielded exact text matches, whereas slightly looser high-IoU crops from Faster R-CNN introduced spurious characters[cite: 1].
*   **Deployment Recommendations:** Faster R-CNN is recommended for accuracy-bounded, single-shot pipelines (e.g., forensic CCTV analysis)[cite: 1]. YOLO26n is recommended for throughput-bounded streaming deployments (e.g., live traffic monitoring)[cite: 1].

---

## 🔬 Architectures & Training Dynamics

Both models were trained on an identical 433-image dataset containing highly heterogeneous, real-world vehicle images[cite: 1]. 

*   **Faster R-CNN:** Configured with a ResNet-50 Feature Pyramid Network (FPN) backbone[cite: 1]. It utilized a custom two-phase training curriculum (head stabilization followed by a differential-learning-rate fine-tune) and employed plate-geometry-aware RPN anchor ratios of `{0.25, 0.5, 1.0}` to compensate for the wide-short aspect ratio typical of license plates[cite: 1].
*   **YOLO26n:** An anchor-free, single-stage model representing realistic out-of-the-box deployment[cite: 1]. It converged effectively over 50 epochs using its native augmentation stack[cite: 1].

<p align="center">
  <img src="../automatic_licence_plate_detection/Project Report images/frcnn_training_curves.png" alt="Faster R-CNN Training Curves" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp;

  <img src="../automatic_licence_plate_detection/Project Report images/yolo26_training_curves.png" alt="YOLO26n Training Curves" width="45%">
</p>

---

## 🎯 Static Detection Quality

Faster R-CNN dominated YOLO26n across aggregate metrics, largely driven by its robustness rather than typical-case tightness. While YOLO26n had a marginally higher median IoU (0.8136 vs. 0.7945), its single-stage architecture resulted in a heavier low-IoU failure tail[cite: 1].

<p align="center">
  <img src="../automatic_licence_plate_detection/Project Report images/model_comparison_iou.png" alt="Model Comparison IoU" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="../automatic_licence_plate_detection/Project Report images/detection_rate_comparison.png" alt="Detection Rate Comparison" width="45%">
</p>

### Sample Predictions (Validation Split)
<p align="center">
  <img src="../automatic_licence_plate_detection/Project Report images/frcnn_sample_predictions.png" alt="Faster R-CNN Sample Predictions" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="../automatic_licence_plate_detection/Project Report images/yolo26_sample_predictions.png" alt="YOLO26n Sample Predictions" width="45%">
</p>

---

## ⚡ Video Inference & Throughput

Benchmarked on an unseen traffic video stream, YOLO26n sustained 41.6 FPS compared to Faster R-CNN's 6.2 FPS[cite: 1]. Despite the speed disparity, both detectors demonstrated excellent and nearly indistinguishable temporal consistency (negligible bounding box jitter across consecutive frames)[cite: 1].

<p align="center">
  <img src="../automatic_licence_plate_detection/Project Report images/video_comparison.png" alt="Video Inference and Temporal Consistency Comparison" width="90%">
</p>

<p align="center">
  <video src="../automatic_licence_plate_detection/Project videos/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1)_frcnn_annotated.mp4" width="100%" controls="controls"></video>
</p>

<p align="center">
  <video src="../automatic_licence_plate_detection/Project videos/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1)_yolo_annotated.mp4" width="100%" controls="controls"></video>
</p>
---

## 🔍 End-to-End Generalization (OCR Integration)

When coupled with EasyOCR, the models demonstrated that a "good" detection is highly sensitive to crop tightness. On an unseen test image, both models agreed on the plate location with an excellent Inter-model IoU of 0.853[cite: 1]. However, Faster R-CNN's slightly looser box included chrome framing that EasyOCR misread as an "I" (`ILETITGO1`), while YOLO26n's tighter crop resulted in a perfect read (`LETITG01`)[cite: 1].

<p align="center">
  <img src="../automatic_licence_plate_detection/Project Report images/generalisation_test_comparison.png" alt="Generalisation Test Comparison with OCR Output" width="90%">
</p>

<p align="center">
  <img src="../automatic_licence_plate_detection/Project Report images/generalisation_test_comparison.png" alt="Generalisation Test Comparison with OCR Output" width="90%">
</p>
---

## 💻 Repository Structure & Execution

### Project File
- Main notebook: `car_licence_plate_detection.ipynb`

### Recommended Environment
This notebook is designed for **Google Colab** with hardware acceleration (GPU).
*   Notebook code natively utilizes `/content` paths.
*   Leverages the upload flow for `kaggle.json` credentials.
*   Model training and video inference heavily rely on GPU availability.

### How To Run
1. Open the notebook in Google Colab.
2. Enable GPU: `Runtime` -> `Change runtime type` -> Select `GPU`.
3. Upload your Kaggle API key file (`kaggle.json`) when prompted by the setup cell.
4. Run all cells sequentially from top to bottom.

*The automated pipeline will download datasets, prepare data splits/labels, train both models, evaluate metrics, and generate all plots, OCR extracts, and video inference outputs.*

### Requirements
Most packages are preinstalled in Google Colab. The notebook handles specific installations natively. If running locally, ensure the following dependencies are met:
*   `python 3.10+`
*   `torch`, `torchvision`
*   `ultralytics`
*   `kaggle`
*   `numpy`, `pandas`, `matplotlib`, `pillow`
*   `albumentations`, `opencv-python`, `easyocr`

### Dataset Credits
*   Training/Validation Images: `andrewmvd/car-plate-detection` (Kaggle)[cite: 1].
*   Video Inference: `nimishshandilya/car-number-plate-video` (Kaggle).

---

## ✍️ Author
**Constantino Harry Alexander**