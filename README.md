# ⚙️ Predictive Maintenance System

> An end-to-end Machine Learning system for predicting **machine failures before they occur**, using industrial sensor measurements and machine operating conditions.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC4E20)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

---

## 📌 Project Overview

Predictive maintenance uses machine learning to identify patterns in equipment data that may indicate an upcoming machine failure.

Instead of waiting for a machine to fail, a predictive maintenance system can analyze operational measurements such as:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine type
* Failure-mode indicators

and estimate whether a **machine failure** is likely.

This project implements a complete binary classification pipeline, starting from exploratory data analysis and data validation and progressing through multiple machine learning models, cross-validation, probability calibration, threshold optimization, error analysis, and model serialization.

---

# 🎯 Project Objectives

The main objectives are to:

* Explore industrial machine sensor data.
* Validate the quality and structure of the dataset.
* Analyze the machine-failure class distribution.
* Investigate numerical feature distributions and correlations.
* Identify and handle irrelevant identifier columns.
* Address severe class imbalance.
* Build reusable preprocessing pipelines.
* Establish multiple baseline classification models.
* Train an XGBoost classifier.
* Compare model performance using multiple evaluation metrics.
* Perform stratified 5-fold cross-validation.
* Calibrate XGBoost probabilities.
* Analyze false positives and false negatives.
* Optimize the classification threshold using F1-score.
* Save the trained model for future predictions.

---

# 🏭 Problem Statement

Machine failures can result in:

* Unexpected downtime
* Production interruptions
* Maintenance costs
* Equipment damage
* Reduced operational efficiency

The goal of this project is to build a binary classification model that predicts:

```text
0 → No Machine Failure
1 → Machine Failure
```

The prediction can be used as an early warning signal for maintenance-related decision support.

---

# 📊 Dataset

The project uses a predictive-maintenance dataset containing:

```text
10,000 rows
14 columns
```

The dataset contains machine identifiers, machine type, operating conditions, failure indicators, and the target variable.

---

# 📋 Dataset Features

| Feature                   | Description                                | Type        |
| ------------------------- | ------------------------------------------ | ----------- |
| `UDI`                     | Unique identifier for each record          | Numerical   |
| `Product ID`              | Product/machine identifier                 | Categorical |
| `Type`                    | Machine/product type                       | Categorical |
| `Air temperature [K]`     | Air temperature in Kelvin                  | Numerical   |
| `Process temperature [K]` | Process temperature in Kelvin              | Numerical   |
| `Rotational speed [rpm]`  | Rotational speed in revolutions per minute | Numerical   |
| `Torque [Nm]`             | Machine torque in Newton-meters            | Numerical   |
| `Tool wear [min]`         | Tool wear duration in minutes              | Numerical   |
| `Machine failure`         | Target variable                            | Binary      |
| `TWF`                     | Tool Wear Failure indicator                | Binary      |
| `HDF`                     | Heat Dissipation Failure indicator         | Binary      |
| `PWF`                     | Power Failure indicator                    | Binary      |
| `OSF`                     | Overstrain Failure indicator               | Binary      |
| `RNF`                     | Random Failure indicator                   | Binary      |

---

# 🎯 Target Variable

The prediction target is:

```text
Machine failure
```

with:

```text
0 → Machine does not fail
1 → Machine fails
```

### Target Distribution

The dataset contains:

```text
No Failure:  9,661
Failure:       339
```

Therefore, machine failure is a **highly imbalanced classification problem**.

Failure cases represent only a small portion of the total observations.

---

# 🔎 Exploratory Data Analysis

The project performs several EDA steps before model development.

## 1. Basic Data Inspection

The dataset is inspected using:

* `head()`
* `tail()`
* `sample()`
* `shape`
* `info()`
* `describe()`

The original dataset contains:

```text
Rows:     10,000
Columns:  14
```

---

## 2. Data Types

The dataset contains:

```text
Numerical columns
Categorical columns
Binary failure indicators
```

The main categorical variables are:

```text
Type
Product ID
```

---

## 3. Missing-Value Analysis

Missing values are checked using:

```python
df.isnull().sum()
```

The dataset contains:

```text
0 missing values
```

across all columns.

---

## 4. Duplicate Analysis

Duplicate records are checked using:

```python
df.duplicated().sum()
```

Result:

```text
0 duplicate rows
```

---

## 5. Class Imbalance Analysis

The target distribution is visualized using a count plot.

The imbalance is significant:

```text
Class 0 → 9661
Class 1 → 339
```

This imbalance is explicitly considered during model development.

---

## 6. Outlier Analysis

Boxplots are generated for the numerical features to inspect their distributions and identify potential extreme observations.

The project does not blindly remove every statistical outlier.

This is important because:

```text
Unusual sensor measurement
        ≠
Invalid measurement
```

In industrial predictive-maintenance data, extreme values can potentially represent genuine operating conditions or failure-related behavior.

---

## 7. Correlation Analysis

A correlation heatmap is generated for numerical variables.

This helps investigate relationships between:

* Temperature
* Rotational speed
* Torque
* Tool wear
* Failure indicators
* Machine failure

---

# 🧹 Data Preparation

After EDA, a copy of the dataset is created:

```python
df_copy = df.copy()
```

Two identifier columns are removed:

```text
UDI
Product ID
```

### Why remove them?

These columns primarily act as identifiers rather than meaningful operating measurements.

The modeling dataset therefore focuses on machine characteristics and operating conditions.

---

# 🧩 Feature and Target Separation

The target is separated from the input features:

```python
X = df_copy.drop(["Machine failure"], axis=1)
y = df_copy["Machine failure"]
```

The resulting features contain:

### Numerical Features

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
TWF
HDF
PWF
OSF
RNF
```

### Categorical Feature

```text
Type
```

---

# ✂️ Train-Test Split

The dataset is divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)
```

Therefore:

```text
Training Data → 67%
Testing Data  → 33%
```

The training set contains:

```text
Class 0 → 6462
Class 1 → 238
```

---

# ⚖️ Handling Class Imbalance

Because machine failures are relatively rare, the model needs to account for the minority class.

The project calculates:

```python
neg, pos = np.bincount(y_train)

scale_weights = neg / pos
```

The resulting class weight is approximately:

```text
27.15
```

This means the positive/failure class receives substantially greater weight during XGBoost training.

---

# ⚙️ Data Preprocessing

The project uses:

* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`

This creates a reproducible preprocessing workflow.

---

## Numerical Preprocessing

For models that require scaling:

```text
Numerical Features
        ↓
Median Imputation
        ↓
StandardScaler
```

For models that do not require scaling:

```text
Numerical Features
        ↓
Median Imputation
```

---

## Categorical Preprocessing

The `Type` feature is processed using:

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

This allows categorical machine types to be converted into numerical representations suitable for machine learning algorithms.

---

# 🤖 Machine Learning Models

The project evaluates multiple classification algorithms.

The models include:

1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine
4. Naive Bayes
5. Decision Tree
6. XGBoost

This provides a broad comparison between:

```text
Linear Models
Distance-Based Models
Kernel-Based Models
Probabilistic Models
Tree-Based Models
Gradient Boosting
```

---

# 1️⃣ Logistic Regression

The Logistic Regression model uses:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

Standard scaling is applied to numerical features.

The `balanced` class-weight option is used to account for the imbalanced target.

---

# 2️⃣ K-Nearest Neighbors

KNN configuration:

```python
KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="euclidean"
)
```

Because KNN is distance-based, numerical features are standardized before training.

---

# 3️⃣ Support Vector Machine

The project uses an RBF-kernel SVM:

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
```

Feature scaling is applied before classification.

---

# 4️⃣ Naive Bayes

The project uses:

```python
GaussianNB()
```

with an unscaled preprocessing pipeline.

---

# 5️⃣ Decision Tree

The Decision Tree configuration is:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    class_weight="balanced",
    random_state=42
)
```

The maximum tree depth is limited to 5.

---

# 6️⃣ XGBoost

XGBoost is used as the primary gradient-boosting model.

The initial XGBoost model uses:

```python
XGBClassifier(
    scale_pos_weight=scale_weights,
    random_state=42
)
```

XGBoost does not require standard feature scaling, so the unscaled preprocessing pipeline is used.

---

# 📊 XGBoost Test Performance

The initial XGBoost model achieved:

| Metric    |        Score |
| --------- | -----------: |
| Accuracy  | **99.8788%** |
| Precision | **98.9899%** |
| Recall    | **97.0297%** |
| F1-score  | **98.0000%** |
| ROC-AUC   | **98.8157%** |

### Confusion Matrix

```text
                 Predicted
                 0       1
Actual 0       3198      1
Actual 1          3     98
```

This corresponds to:

```text
True Negatives  = 3198
False Positives = 1
False Negatives = 3
True Positives  = 98
```

---

# 🔄 Stratified 5-Fold Cross-Validation

To obtain a more robust estimate of model performance, the project uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The same folds preserve the approximate class distribution across each validation split.

---

# 📈 Cross-Validation Results

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Results

| Model               | Accuracy | Precision |   Recall |       F1 |  ROC-AUC |
| ------------------- | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression | 0.999104 |  1.000000 | 0.974645 | 0.987046 | 0.981053 |
| KNN                 | 0.998955 |  1.000000 | 0.970479 | 0.984895 | 0.985168 |
| SVM                 | 0.999104 |  1.000000 | 0.974645 | 0.987046 | 0.979732 |
| Naive Bayes         | 0.998060 |  0.971376 | 0.974645 | 0.972745 | 0.989649 |
| Decision Tree       | 0.997313 |  0.962034 | 0.961968 | 0.961960 | 0.980288 |
| XGBoost             | 0.998955 |  0.995918 | 0.974645 | 0.984984 | 0.988426 |

These results show that several models perform strongly on this dataset, which makes cross-validation particularly useful for understanding model behavior beyond a single train-test split.

---

# 🎯 Why Multiple Evaluation Metrics?

Accuracy alone can be misleading for highly imbalanced datasets.

For example, because machine failures are relatively rare, a model could achieve high accuracy while missing many actual failures.

Therefore, the project evaluates:

### Precision

Measures how many predicted failures were actually failures.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual failures were detected.

```text
Recall = TP / (TP + FN)
```

### F1-score

Balances precision and recall.

```text
F1 = 2 × Precision × Recall
      -----------------------
       Precision + Recall
```

### ROC-AUC

Measures the ability of the model to distinguish between failure and non-failure cases across thresholds.

---

# 🎯 Probability Calibration

The project applies probability calibration to the XGBoost model using:

```python
CalibratedClassifierCV(
    estimator=xgb_pipeline,
    method="sigmoid",
    cv=5
)
```

The workflow compares:

```text
Uncalibrated XGBoost probabilities
              vs
Calibrated XGBoost probabilities
```

using calibration curves.

The objective is to make predicted probabilities more representative of the observed frequency of machine failures.

---

# ❌ False Positive & False Negative Analysis

After calibration, the model generates failure probabilities:

```python
y_prob = calibrated_xgb.predict_proba(X_test)[:, 1]
```

The default classification threshold is initially:

```text
0.50
```

Predictions are then compared against actual labels.

---

## False Positive

A false positive occurs when:

```text
Actual = 0
Predicted = 1
```

The model predicts a machine failure when no failure occurred.

---

## False Negative

A false negative occurs when:

```text
Actual = 1
Predicted = 0
```

The model fails to identify an actual machine failure.

---

## Error Analysis Results

The notebook reports:

```text
False Positives: 0
False Negatives: 3
```

The project explicitly extracts these observations for further inspection.

This is particularly relevant in predictive maintenance because missed failures can represent an important operational risk.

---

# 🎚️ Threshold Optimization

The default threshold of:

```text
0.50
```

is not necessarily optimal for every classification problem.

The project uses the Precision-Recall relationship to search for a threshold that maximizes the F1-score.

```python
precisions, recalls, thresholds = precision_recall_curve(
    y_test,
    calibrated_proba
)
```

F1-scores are calculated for the candidate thresholds.

The best threshold obtained in the notebook is:

```text
0.6712224544891623
```

Therefore, the final prediction logic can be represented as:

```python
prediction = (probability >= 0.6712224544891623).astype(int)
```

This demonstrates an important machine-learning concept:

> The probability threshold can be optimized according to the desired balance between precision and recall rather than always using 0.50.

---

# 💾 Model Serialization

The trained calibrated model is saved using `joblib`.

### Saved Model

```text
Predictive Maintaince System.pkl
```

### Saved Threshold

```text
Best_Threshold.pkl
```

These files allow the trained system to be reused without retraining the model.

---

# 🔮 Making Predictions

The saved model can be loaded using:

```python
import joblib

model = joblib.load("Predictive Maintaince System.pkl")
threshold = joblib.load("Best_Threshold.pkl")
```

For new machine data:

```python
probability = model.predict_proba(X_new)[:, 1]

prediction = (probability >= threshold).astype(int)
```

Interpretation:

```text
0 → No predicted machine failure
1 → Predicted machine failure
```

The probability can additionally be used as a risk signal.

---

# 🧠 Complete Machine Learning Workflow

```text
                  ┌──────────────────────┐
                  │  Predictive Dataset  │
                  │      10,000 Rows     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │         EDA          │
                  ├──────────────────────┤
                  │ Data Types           │
                  │ Missing Values       │
                  │ Duplicates           │
                  │ Class Distribution   │
                  │ Outliers             │
                  │ Correlation          │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Preparation     │
                  │                      │
                  │ Remove UDI           │
                  │ Remove Product ID    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature / Target     │
                  │ Separation           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Train/Test Split     │
                  │      67 / 33         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Class Imbalance      │
                  │ scale_pos_weight     │
                  │       ≈ 27.15        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Preprocessing        │
                  │                      │
                  │ Imputation           │
                  │ Scaling              │
                  │ One-Hot Encoding     │
                  └──────────┬───────────┘
                             │
                             ▼
             ┌───────────────┴────────────────┐
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │  Baseline Models│              │     XGBoost     │
    ├─────────────────┤              ├─────────────────┤
    │ Logistic Reg.   │              │ Gradient Boost. │
    │ KNN             │              │ Class Weight    │
    │ SVM             │              └────────┬────────┘
    │ Naive Bayes     │                       │
    │ Decision Tree   │                       ▼
    └────────┬────────┘              ┌─────────────────┐
             │                       │ 5-Fold Stratified│
             │                       │ Cross Validation │
             │                       └────────┬────────┘
             │                                │
             │                                ▼
             │                       ┌─────────────────┐
             │                       │ Probability     │
             │                       │ Calibration     │
             │                       └────────┬────────┘
             │                                │
             │                                ▼
             │                       ┌─────────────────┐
             │                       │ Error Analysis  │
             │                       │ FP / FN         │
             │                       └────────┬────────┘
             │                                │
             │                                ▼
             │                       ┌─────────────────┐
             │                       │ Threshold       │
             │                       │ Optimization    │
             │                       └────────┬────────┘
             │                                │
             └────────────────┬───────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Model Evaluation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Saved ML Model   │
                    │ + Threshold      │
                    └──────────────────┘
```

---

# 🛠️ Technology Stack

## Programming Language

* Python

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Model Calibration

* `CalibratedClassifierCV`
* `calibration_curve`

## Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve

## Model Persistence

* Joblib

## Development Environment

* Jupyter Notebook
* Google Colab

---

# 📁 Recommended Repository Structure

```text
predictive-maintenance-system/
│
├── 📓 Predictive Maintenance System.ipynb
│
├── 📊 data/
│   └── predictive_maintenance.csv
│
├── 🤖 models/
│   ├── Predictive Maintaince System.pkl
│   └── Best_Threshold.pkl
│
├── 📈 images/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_boxplots.png
│   ├── confusion_matrix.png
│   └── calibration_curve.png
│
├── 📄 README.md
│
└── 📦 requirements.txt
```

> **Note:** The notebook currently saves the model using the filename `Predictive Maintaince System.pkl`. The spelling can be cleaned up to `Predictive_Maintenance_System.pkl` before publishing if you want a more professional repository structure.

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/predictive-maintenance-system.git
```

Navigate into the project:

```bash
cd predictive-maintenance-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
jupyter
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Predictive Maintenance System.ipynb
```

Run the notebook sequentially.

The notebook follows:

```text
EDA
→ Data Validation
→ Preprocessing
→ Model Training
→ Cross-Validation
→ Calibration
→ Error Analysis
→ Threshold Optimization
→ Model Saving
```

---

# 📊 Key Results

### Dataset

```text
10,000 records
14 original columns
339 machine failures
9,661 non-failure records
0 missing values
0 duplicate records
```

### Class Imbalance

```text
Training Class 0: 6462
Training Class 1:  238

scale_pos_weight ≈ 27.15
```

### XGBoost Test Results

```text
Accuracy  : 99.8788%
Precision : 98.9899%
Recall    : 97.0297%
F1 Score  : 98.0000%
ROC-AUC   : 98.8157%
```

### Confusion Matrix

```text
[[3198    1]
 [   3   98]]
```

### Calibrated Model Error Analysis

```text
False Positives: 0
False Negatives: 3
```

### Optimized Threshold

```text
0.6712224544891623
```

---

# 🔬 Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Exploratory Data Analysis
* Data validation
* Missing-value checking
* Duplicate detection
* Outlier analysis
* Correlation analysis
* Feature selection
* Identifier removal
* Train-test splitting
* Class imbalance
* Class weighting
* Data preprocessing
* Numerical imputation
* Categorical imputation
* One-hot encoding
* Standardization
* Scikit-learn pipelines
* ColumnTransformer
* Logistic Regression
* KNN
* SVM
* Naive Bayes
* Decision Trees
* XGBoost
* Stratified K-Fold Cross-Validation
* Probability calibration
* Confusion matrix analysis
* False-positive analysis
* False-negative analysis
* Precision-Recall curves
* Threshold optimization
* Model serialization

---

# 🚀 Future Improvements

The current project can be extended into a production-style predictive maintenance application.

## 1. Real-Time Prediction API

Build a REST API using:

```text
FastAPI
```

Example endpoint:

```text
POST /predict
```

The API could accept machine sensor readings and return:

```json
{
  "failure_probability": 0.82,
  "prediction": 1
}
```

---

## 2. Interactive Dashboard

Build a monitoring dashboard using:

```text
Streamlit
```

Possible dashboard components:

* Machine status
* Failure probability
* Sensor readings
* Maintenance alerts
* Historical predictions
* Model confidence
* Failure trends

---

## 3. Real-Time Monitoring

Connect the model to simulated or real sensor streams:

```text
Machine Sensors
      ↓
Data Ingestion
      ↓
Preprocessing
      ↓
ML Model
      ↓
Failure Probability
      ↓
Maintenance Alert
```

---

## 4. Model Explainability

Future versions can include:

* SHAP
* Feature importance
* Local prediction explanations
* Failure-driver analysis

This can help identify which machine measurements contributed most to a prediction.

---

## 5. MLOps

The project could be extended with:

```text
MLflow
Docker
GitHub Actions
FastAPI
Model Registry
Monitoring
CI/CD
```

---

## 6. Cost-Sensitive Threshold Optimization

The current threshold is optimized using F1-score.

A production system could instead define different costs for:

```text
False Positive
False Negative
```

and optimize the threshold according to the actual maintenance/business objective.

---

## 7. Model Monitoring

A production system should monitor:

* Data drift
* Feature drift
* Prediction drift
* Failure-rate changes
* Model performance
* Calibration
* Missing values
* Sensor anomalies

---

# ⚠️ Limitations

Although the model achieves strong performance on the provided dataset, several considerations are important.

### Dataset Imbalance

Machine failures represent a relatively small portion of the dataset.

Therefore, accuracy alone should not be used to judge the system.

### Dataset Size

The dataset contains 10,000 observations, which may not represent all real-world industrial environments.

### Generalization

Performance on this dataset does not guarantee the same performance on:

* Different machines
* Different factories
* Different sensors
* Different operating environments
* Future production data

### Production Deployment

The current project is a machine-learning portfolio implementation and is not a complete production monitoring system.

A production implementation would require additional:

* Data pipelines
* Monitoring
* Reliability testing
* Model governance
* Security
* Sensor integration
* Retraining strategy

---

# 💡 Why Predictive Maintenance?

Traditional maintenance approaches include:

```text
Reactive Maintenance
        ↓
Machine fails
        ↓
Repair
```

Predictive maintenance attempts to move toward:

```text
Machine Sensors
        ↓
Continuous Data
        ↓
Machine Learning
        ↓
Failure Probability
        ↓
Early Warning
        ↓
Planned Maintenance
```

This approach can support maintenance teams by identifying potentially risky machine conditions before an observed failure.

---

# 📌 Project Highlights

### End-to-End ML Pipeline

```text
EDA
→ Data Validation
→ Feature Preparation
→ Class Imbalance Handling
→ Preprocessing
→ Model Training
→ Cross-Validation
→ Calibration
→ Error Analysis
→ Threshold Optimization
→ Model Serialization
```

### Multiple Algorithms

```text
Logistic Regression
KNN
SVM
Naive Bayes
Decision Tree
XGBoost
```

### Advanced ML Components

```text
Stratified 5-Fold CV
Probability Calibration
False Positive Analysis
False Negative Analysis
Precision-Recall Threshold Optimization
Model Persistence
```

---

# 👨‍💻 Author

## Muhammad Ibrahim

**Software Engineering | Machine Learning | AI**

📍 Pakistan

---

# ⭐ Acknowledgements

This project was developed as part of practical machine-learning learning and portfolio development, with a focus on applying classification techniques to an industrial predictive-maintenance problem.

---

# 📄 License

This project is available under the **MIT License**.
