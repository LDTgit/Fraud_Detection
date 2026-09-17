# Fraud Detection using Machine Learning

This project focuses on building a machine learning model to detect fraudulent transactions in a highly imbalanced financial dataset. The goal is to identify anomalies and potentially fraudulent behavior, minimizing missed fraud cases while also managing false alarms.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Setup and Installation](#setup-and-installation)
4.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    *   [Feature Engineering](#feature-engineering)
    *   [Model Training](#model-training)
    *   [Model Optimization](#model-optimization)
5.  [Results and Evaluation](#results-and-evaluation)
6.  [Key Findings](#key-findings)


## Project Overview

The financial industry constantly battles sophisticated fraud schemes. This project addresses the challenge of identifying fraudulent transactions, especially within the datasets where fraudulent cases are rare compared to legitimate ones. We employ a Random Forest Classifier, alongside robust data preprocessing, exploratory data analysis and crucial model optimization techniques to achieve a balanced detection rate.

## Dataset

The dataset used for this project was downloaded from Kaggle: [Fraud Detection Dataset] (https://www.kaggle.com/datasets/waddahali/fraud-detection)
It contains various features related to financial transactions and a target variable (`is_fraud`) indicating whether a transaction is legitimate (0) or fraudulent (1). A significant characteristic of this dataset is its **high imbalance**, with only approximately 10% of transactions being fraudulent.

## Setup and Installation

To run this notebook, you'll need a Google Colab environment or a local Python environment with the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**To get started:**
1. **Download the Dataset:** Obtain the `fraud.csv` file from the Kaggle link provided above.
2. **Upload to Google Colab:** If using Google Colab, upload the `fraud.csv` to your session storage using the `Files` tab in the left sidebar.
3. **Run the Notebook:** Execute the cells sequentially in the provided Jupyter/Colab notebook.

 ## Methodology
 ### Data Preprocessing
 * **Handling Missing Values:** Missing values were identified across several columns. Categorical features (`is_weekend`, `device_type`, `store_type`) were imputed using their mode.
`prev_transactions` missing values were replaced with 0.
The `is_first_transaction` column was imputed based on `prev_transactions`.
Remaining numerical missing values were imputed using the median.
 *  **Data Imbalance:** The inherent imbalance in the target variable (`is_fraud`) was noted early in the process.

### Exploratory Data Analysis (EDA)
* **Dataset Structure:** Verified the number of rows and columns and data types.
* **Descriptive Statistics:** Generated summary statistics for all features.
* **Attribute Distribution:** Visualized the distribution of both the target variable and descriptive attributes using count plots and histograms.
* **Correlation Analysis:** A correlation matrix was generated to understand relationships between features. Notably, an examination of the correlation matrix reveals a relationship between the engineered relationship between the engineered `poisson_anomaly` feature and `velocity_score`, from which it is partially derived. While some correlation exists, for tree-based models like Random Forests, this collinearity typically does not negatively impact predictive performance. Instead, it can influence the interpretation of individual feature importance, as the predictive 'credit' might be distributed between correlated features. However, `poisson_anomaly` provides a distinct measure of statistical unusualness that complements the raw `velocity_score`.
<img width="200" alt="image" src="https://github.com/LDTgit/Fraud_Detection/blob/main/correlation_matrix.png" />

### Feature Engineering
A new feature, `poisson_anomaly` was engineered to capture anomalous transaction frequencies. This feature leverages the Poisson distribution to asses the probability of a given number of recent transactions (`velocity_score`) occurring based on the historical average transaction rate (`prev_transactions`) for specific age groups (`customer_age`). A lower `poison_anomaly` score indicates a higher statistical anomaly, potentially pointing to fraudulent behavior. This aims to enhance the model's ability to detect unusual patterns that might not be obvious from raw features.

### Model Training
A `RandomForestClassifier` was chosen for its robustness and ability to handle imbalanced datasets (using `class_weight = 'balanced'`). The dataset was split into training and testing sets (80/20 ratio) with stratification to maintain the class distribution.

### Model Optimization
To improve the model's performance, especially in detecting the minority class (fraud), two main optimization techniques were applied:
1. **Hyperparameter Tuning (Grid Search):** `GridSearchCV` was used to find the optimal hyperparameters for the `RandomForestClassifier`. The search focused on `n_estimators`, `max_depth` and `min_samples_split`, with the `f1-score` as the scoring metric, which in more suitable for imbalanced datasets than accuracy.
2. **Prediction Threshold Adjustment:** After identifying the best model from Grid Search, the prediction threshold was fine-tuned. By analyzing the `precision_recall_curve`, an optimal threshold was automatically determined that maximized the F1-score for the fraud class. This step is crucial for balancing the trade-off between identifying more frauds (recall) and minimizing false positives (precision) in an imbalanced scenario.

## Results and Evaluation
The model's performance was evaluated using:
* **Confusion Matrix:** To understand True Positives, True Negatives, False Positives and False Negatives.
<img width="200" alt="image" src="https://github.com/LDTgit/Fraud_Detection/blob/main/correlation_matrix.png" />
* **Classification Report:** Providing Precision, Recall and F1-Score for each class.
* **Accuracy:** Overall correctness of predictions.
* **ROC Curve and AUC:** Visualizing the trade-off between True and Positive Rate and False Positive Rate.
<img width="200" alt="image" src="https://github.com/LDTgit/Fraud_Detection/blob/main/ROC.png" />

**Initial Model Performance (before optimization):**
* Recall (fraud class): ~28%
* Accuracy: ~68%

**Optimized Model Performance (after Grid Search and Threshold Adjustment):**
* Recall (fraud class): ~39%
* Accuracy: ~66%

While the overall accuracy sightly decreased, the **Recall for the fraud class significantly improved from ~28% to 39%.** This indicates the optimized model is better at catching fraudulent transactions, which is a primary goal in fraud detection. The precision for the fraud class, though still low at ~13%, is acceptable given the extreme data imbalance.

## Key Findings
* **Data Imbalance is Critical:** The dataset's severe imbalance (only ~10% fraud) necessitated specialized techniques like `class_weight='balanced'` and `f1-score` based evaluation and optimization.
* **Effective Feature Engineering:** The `poission_anomaly` feature, which quantifies the unusual frequency of transactions, proved to be a strong predictor, ranking highly in feature importance.
* **Importance of Optimization:** Both hyperparameter tuning and prediction threshold adjustment were essential for improving the detection of fraudulent transactions. The threshold adjustment specifically allowed for a better balance between catching fraud and minimizing false alarms.
* **Top Predictors:** `distance_from_home`, `velocity_score`, `network_quality`, `transaction_amount` and the engineered `poisson_anomaly` were identified as the most important features in detecting fraud. 
