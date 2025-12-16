---
title: MLOps Lab2 GUI
emoji: 🐶
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---
# 🐶 MLOps Lab 3: Pet Breed Classifier

![CI Status](https://github.com/andresmln/MLOps-Lab3/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Pylint](https://img.shields.io/badge/pylint-10.00-brightgreen)

**Author:** Andrés Malón  
**Master:** Machine Learning   
**Date:** December 2025

## 🚀 Deployed Services
Here are the links to the running application:

| Service | Status | Link |
|:-------:|:------:|:----:|
| **API (Render)** | 🟢 Online | [➡️ Swagger UI / Docs](https://mlops-lab2-api-ayg6.onrender.com) |
| **Frontend (Hugging Face)** | 🟢 Online | [➡️ Try the App](https://huggingface.co/spaces/andresmln/MLOps-Lab2-GUI) |

## 📖 Project Description
This project implements a complete **MLOps pipeline** for classifying pet breeds using **MobileNetV2** (Transfer Learning). It includes:
* **Model:** MobileNetV2 trained on Oxford-IIIT Pet Dataset (37 classes).
* **API:** Built with FastAPI and ONNX Runtime.
* **CI/CD:** Automated testing (Pytest) and Linting (Pylint) via GitHub Actions.
* **Deployment:** Auto-deploy to Render (Backend).

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/andresmln/MLOps-Lab3.git](https://github.com/andresmln/MLOps-Lab3.git)
   ```
2. **Create and activate a virtual environment:**
# Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
# Mac/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
4. **Run the API**
```bash
python api/api.py
```

## ✅ Quality Assurance
* **Tests:** 100% passing.
* **Linter:** 10.00/10.


