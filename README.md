# ✍️ Linking Writing Processes to Writing Quality (Kaggle Solution)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-enabled-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-enabled-yellow)

This repository contains a solution for the Kaggle competition [Linking Writing Processes to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality). 

The main objective of the competition is to predict the final quality score of an essay based on the logs of its writing process (keystrokes, mouse movements, time delays).

## 🚀 Approach Overview

The solution is based on generating features from user input time series and utilizing an ensemble of gradient boosting algorithms.

### 1. Feature Engineering
The raw data consists of event logs, which are transformed into a tabular dataset for machine learning:
* **Time Series Aggregations:** Calculating `min`, `max`, `mean`, `std`, `sum`, and `median` for key hold times (`down_time`, `up_time`), action durations (`action_time`), text length (`word_count`), and cursor positions.
* **Activity Counts (`activity`):** Counting the frequency of each action type (e.g., *Input*, *Remove/Cut*, *Nonproduction*).
* **Specific Keystroke Counts:** Counting the number of times crucial keys were pressed, reflecting the typing and editing process (Space, Backspace, Shift, Enter, and punctuation marks).

### 2. Data Preprocessing
* **Feature Synchronization:** Using `align` / `reindex` to strictly match the columns of the test set with the training set. This solves the issue of missing events in the `test_logs.csv` dataset.
* **Column Name Cleaning:** Removing special characters and converting column names to the `[A-Za-z0-9_]` format, which is critical to prevent errors within LightGBM.

### 3. Models and Validation
The primary algorithm is a blend (ensemble) of two models:
* **LightGBM Regressor** (`learning_rate=0.02`, `max_depth=5`)
* **CatBoost Regressor** (`learning_rate=0.02`, `depth=5`)

To evaluate the models' performance and prevent overfitting, **10-Fold Cross-Validation** is applied. The final prediction (`score`) is calculated as the arithmetic mean of the predictions from both models.

## 📂 Repository Structure

* `kaggle_solution_clean_names.ipynb` — A Jupyter Notebook containing the full solution pipeline: from data loading and processing to model training and generating the final prediction file.

## ⚙️ Environment Requirements

To run the code, you will need the following libraries:

```bash
pip install pandas numpy lightgbm catboost scikit-learn
