# Machine Learning Models Project

This repository contains machine learning models for classification tasks, including models for heart disease prediction and other applications.

## Project Structure

- **bayes.ipynb**: Jupyter notebook for training and evaluating the Naive Bayes model
- **decision.ipynb**: Jupyter notebook for training and evaluating the Decision Tree 
- **ANN.ipynb**: Jupyter notebook for training and evaluating the ANN model
- **decision_tree_model.pkl**: Saved Decision Tree model (pickle format)
- **heart_disease_model_new.h5**: Saved heart disease prediction model by ANN (HDF5 format)
- **naive_bayes_model.pkl**: Saved Naive Bayes model (pickle format)

## Model Files

The `.pkl` and `.h5` files are saved trained models that can be loaded directly for inference without needing to retrain. This saves computation time when deploying or testing the models.

## Usage

### Notebook Files

The `.ipynb` files contain the code used to:
1. Load and preprocess data
2. Train the models
3. Evaluate model performance
4. Save the trained models

Use these notebooks if you need to retrain models with new data or adjust model parameters.

## Requirements

To run the notebooks and load the models, you'll need:

- Python 3.x
- Jupyter
- scikit-learn
- TensorFlow/Keras 
- NumPy
- Pandas

Install dependencies using pip:
``` bash
pip install -r requirements.txt
```