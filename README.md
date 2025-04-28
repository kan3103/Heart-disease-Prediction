# Heart Disease Prediction

This is a final project for the **Machine Learning** course, completed by the following team members:

- Lý Nguyên Khang
- Bùi Thanh Bách


---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Notes](#notes)
- [Contact](#contact)

---

## 📝 Project Overview

The project builds a heart disease prediction system using Machine Learning models. The workflow includes:

- Collecting and storing the dataset
- Processing and normalizing the data
- Training and saving machine learning models
- Deploying a web application using **Streamlit** for predictions

---

## ⚙️ Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/kan3103/Heart-disease-Prediction.git
cd Heart-disease-Prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Activate the environment
source venv/bin/activate    # On Linux/macOS
venv\Scripts\activate       # On Windows
```

### 3.Install required libraries

```bash
pip install -r requirements.txt
```

## Project Structure

``` bash
Heart-disease-Prediction/
├── data/
│   └── handle.ipynb            #handle preprocessing data
├── datasets/
│   └── heart_disease.csv       #dataset
├── models/
│   ├── bayes.ipynb                #implement bayes
│   ├── decision.ipynb          # implement dicision tree
│   ├── decision_tree_model.pkl    #save the model by dicision tree
│   ├── heart_disease_model_new.h5  # save the model by ANN
│   ├── ann.ipynb              # implement ANN
│   └── naive_bayes_model.pkl # save the model by naive bayes
├── scaler/
│   └── scaler.pkl # save to scale the input
├── .gitignore
├── GUI.py          # build website to predict
├── README.md
└── requirements.txt

```
## How to run

### 1. Train the Models

If you want to retrain the models:

- Navigate to the `models/` directory.
- Open and run the corresponding `.ipynb` files (`bayes.ipynb`, `decision.ipynb`) to train the models manually.
- After training, the new model files (e.g., `.pkl`, `.h5`) will be saved automatically in the `models/` folder.

> **Note**: Make sure that the dataset (`heart_disease.csv`) and scaler (`scaler.pkl`) are available before training.

### 2. Launch the Web Application
Start the Streamlit web application:
```bash
streamlit run gui.py
```
After running, open your browser and navigate to the provided local URL (typically http://localhost:8501) to access the application.
## Notes

- Ensure that the heart.csv dataset is placed inside the `datasets/` directory.
- You can adjust the model selection or prediction logic in the `GUI.py` file if needed.
- If any library is missing, manually install it using pip.