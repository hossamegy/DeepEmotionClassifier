# Sentiment Analysis and Emotion Classification

## Project Overview

This project aims to perform sentiment analysis and emotion classification using deep learning techniques. It involves several stages, including data importing, cleaning, preprocessing, model training, and evaluation.

## Table of Contents

1. [Usage](#file-structure)
   - [Requirements](#requirements)
   - [Data](#data)
   - [Data Importer](#data-importer)
   - [Data Cleaner](#data-cleaner)
   - [Data Preprocessor](#data-preprocessor)
   - [Model Trainer](#model-trainer)
   - [Model Evaluator](#model-evaluator)
   - [Main Script](#main-script)
    - [Jupyter Notebook](#jupyter-notebook)

2. [License](#license)


## Usage
### Requirements

- **File**: `requirements.txt`
- **Purpose**: This file lists all the dependencies required to run the project, including libraries for data handling, text processing, machine learning, and plotting.

```bash
pip install -r requirements
```

### Data

- **File**: `train.csv`
- **Purpose**: This CSV file contains the raw data used for sentiment analysis and emotion classification. The dataset typically includes:
  - **Text Column**: Contains the text data that will be analyzed. Each row in this column represents a distinct piece of text, such as a sentence or a review.
  - **Target Column**: Contains the labels associated with each piece of text. These labels represent the sentiment or emotion conveyed in the text, such as positive, negative, or neutral sentiments, or various emotional states like happiness, sadness, etc.

- **Data Format**:
  - The CSV file should have at least two columns: one for the text data and one for the target labels.
  - Example:
    ```
    text,target
    "I love this product!",positive
    "This is the worst experience ever.",negative
    "I'm feeling great today.",happy
    ```

### Data Importing

- **File**: `data_importer.py`
- **Function**: This file contains a function to import data from a CSV file into a pandas DataFrame. It prints a success message along with a preview of the first few rows of the dataset to ensure that the data has been loaded correctly.

### Data Cleaning

- **File**: `data_cleaner.py`
- **Function**: This file includes a function to clean and preprocess the text data in a DataFrame. It performs several operations such as converting text to lowercase, removing special characters, numbers, emojis, single characters, and extra spaces. It also removes stopwords except for crucial negation words and rare words that appear fewer than a specified number of times. After cleaning, it prints a success message and shows a preview of the cleaned dataset.

### Data Preprocessing

- **File**: `data_preprocessing.py`
- **Function**: This file provides a function to preprocess the text data by tokenizing it and converting it into padded or truncated sequences. It also encodes the target labels for classification. The preprocessed data is then saved to files for use in model training. The function prints details about the processed data, including label classes and the structure of input sequences.

### Model Training

- **File**: `model_trainer.py`
- **Function**: This file includes a function to train a TensorFlow/Keras model using the preprocessed data. It incorporates callbacks to adjust the learning rate and stop training early if necessary. After training, the model is saved to a specified file. The function prints progress updates and success messages throughout the training process.

### Model Evaluation

- **File**: `model_evaluator.py`
- **Function**: This file provides a function to evaluate the trained model on test data. It prints the evaluation results, including metrics such as accuracy, loss, precision, and recall. The function also plots the training history, showing accuracy and loss over epochs for both training and validation datasets.

### Main Script

- **File**: `trainParallelArchit.py`
- **Function**: This script is used to define model parameters, call the necessary functions for importing, cleaning, preprocessing data, and training and evaluating the model. It orchestrates the entire workflow from data loading to model evaluation.

### Jupyter Notebook

- **File**: `CNN_LSTM_Model.ipynb`
- **Purpose**: This Jupyter Notebook provides an interactive environment to run all the code required for sentiment analysis and emotion classification. It includes cells to:
  
  
  The notebook is designed to provide a step-by-step guide through the entire process, allowing for easy experimentation and visualization of results.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
