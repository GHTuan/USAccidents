# Analysis and Evaluation of Traffic Accident Severity Prediction Models

## Description

This project performs an in-depth analysis of the "US Accidents" dataset (covering traffic accidents in the US from 2016-2023) to build and evaluate machine learning models capable of predicting accident severity. The primary goal is to identify contributing factors and find the most effective model for classifying accidents into two categories: "Less Severe" and "Severe," with a special emphasis on accurately detecting severe cases (optimizing Recall and F1-score).

## Key Features

*   **Data Loading & Selection:** Loading data from a large CSV file and selecting potentially relevant features.
*   **Extensive Preprocessing:**
    *   Handling missing values (dropping columns/rows based on missing rates).
    *   Time feature engineering (extracting year, month, hour, day of week).
    *   Categorical feature simplification (grouping `Weather_Condition`, `Wind_Direction`).
    *   Removing redundant/low-information features (rare POIs, `Wind_Chill(F)` due to high correlation).
*   **Exploratory Data Analysis (EDA):**
    *   Analyzing the target variable (`Severity`) distribution and defining the binary classification task.
    *   Visualizing accident trends over time, geography, and weather conditions.
    *   Analyzing correlations between numerical features.
*   **Model Training & Evaluation:**
    *   Implementing and comparing various algorithms: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM (with/without SMOTE), MLP (Keras).
    *   Applying techniques for handling class imbalance (Class Weighting, SMOTE).
    *   Evaluating models based on Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
*   **Model Selection:** Selecting the best-performing model based on prioritized metrics (Recall, F1-score).

## Models Used

*   Logistic Regression
*   Naive Bayes (GaussianNB)
*   Decision Tree Classifier
*   Random Forest Classifier
*   XGBoost (XGBClassifier)
*   LightGBM (LGBMClassifier)
*   Multi-Layer Perceptron (MLP using Keras/TensorFlow)

## Technologies Used

*   Python 3.x
*   Jupyter Notebook
*   Pandas
*   NumPy
*   Scikit-learn (for preprocessing, modeling, evaluation)
*   Matplotlib & Seaborn (for visualization)
*   Imbalanced-learn (for SMOTE)
*   XGBoost
*   LightGBM
*   TensorFlow / Keras (for MLP)

## Dataset

*   The dataset used is the publicly available "US Accidents" dataset, recording accidents in 49 US states from February 2016 to March 2023.
*   The original data contains detailed information about the time, location, environmental conditions, and traffic characteristics at the time of the accident.
*   **Download:** Please download the original dataset from the source provided in the `data/download.txt` file and place the downloaded CSV file into the `data/` directory.
    *(Note: Due to its large size, the dataset file is not included directly in this repository.)*

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GHTuan/USAccidents.git
    cd USAccidents
    ```
2.  **Download the dataset:**
    *   Refer to `data/download.txt` for the data source link.
    *   Download the `US_Accidents_March23.csv` file (or similar name) and place it inside the `data/` directory.
3.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the analysis:**
    *   Open and execute the cells within the Jupyter Notebook:
        ```bash
        jupyter notebook "Traffic Accidents - Data Mining.ipynb"
        ```

## Project Structure
```bash
USAccidents/
├── data/
│ └── download.txt # Instructions/link to download the dataset
├── requirements.txt # List of required Python packages
└── Traffic Accidents - Data Mining.ipynb # Main Jupyter Notebook with all code and analysis
```

## Results Summary

*   Analysis revealed a significant class imbalance in the target variable `Severity`.
*   Boosting models (XGBoost, LightGBM) and Decision Trees, when using class weights to handle imbalance, demonstrated the best ability to detect severe accidents (highest Recall).
*   **XGBoost** (with `scale_pos_weight`) was identified as the most effective model, achieving the best balance between Recall (highest at ~83.8%) and F1-score (highest at ~0.552).
*   Using SMOTE (with LightGBM) improved Accuracy/Precision but significantly reduced Recall, making it less suitable given the problem's priority.

## Future Work

*   Perform detailed Hyperparameter Tuning for the top-performing models (especially XGBoost).
*   Explore more complex model architectures (e.g., deeper neural networks, advanced ensemble techniques).
*   Integrate additional data sources (e.g., real-time traffic density, detailed road infrastructure characteristics).