# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster using passenger demographic and travel information. This project demonstrates an end-to-end data science workflow, including data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and interpretability. It compares six different classification algorithms: **Random Forest, Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbours (KNN), Gradient Boosting, and XGBoost**.

## 📌 Project Overview

This project uses the Titanic dataset provided through the Seaborn library to build classification models for predicting passenger survival.

The workflow includes:

* Loading and exploring the Titanic dataset
* Selecting relevant passenger features
* Checking the distribution of the target classes
* Splitting the dataset into training and testing sets
* Handling missing numerical and categorical values
* Scaling numerical features
* One-hot encoding categorical features
* Building a preprocessing and classification pipeline
* Hyperparameter tuning using GridSearchCV
* Evaluating model performance using classification reports and confusion matrices
* Analysing feature importance
* Comparing Random Forest and Logistic Regression models

## 📁 Notebook Structure
The project follows a structured approach from data ingestion to final conclusions:

```text
Titanic Survival Prediction
│
├── 1. Import Libraries
│
├── 2. Load Dataset
│
├── 3. Exploratory Data Analysis
│   ├── Dataset overview
│   ├── Missing values
│   └── Target distribution
│
├── 4. Feature Selection
│
├── 5. Train-Test Split
│
├── 6. Data Preprocessing
│   ├── Numerical pipeline
│   ├── Categorical pipeline
│   └── ColumnTransformer
│
├── 7. Model Training
│   ├── Random Forest
│   ├── Logistic Regression
│   ├── Support Vector Machine
│   ├── K-Nearest Neighbours
│   ├── Gradient Boosting
│   └── XGBoost
│
├── 8. Model Evaluation
│   ├── Classification Reports
│   ├── Confusion Matrices
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   └── F1 Score
│
├── 9. ROC-AUC Analysis
│   └── Combined ROC Curves
│
├── 10. Model Comparison
│   ├── Performance table
│   ├── Accuracy comparison
│   └── F1 comparison
│
├── 11. Cross-Validation Analysis
│
├── 12. Feature Importance
│   ├── Random Forest
│   └── XGBoost
│
└── 13. Final Conclusions
```

## 🎯 Objective

The primary objective is to develop a classification model capable of predicting whether a Titanic passenger survived based on available passenger information.

The target variable is:

* `survived` — whether the passenger survived (`1`) or did not survive (`0`)

## 📊 Dataset

The project uses the Titanic dataset available through `seaborn`.

```python
titanic = sns.load_dataset('titanic')
```

### Features Used

The following features were selected for model training:

| Feature      | Description                                |
| ------------ | ------------------------------------------ |
| `pclass`     | Passenger class                            |
| `sex`        | Passenger sex                              |
| `age`        | Passenger age                              |
| `sibsp`      | Number of siblings/spouses aboard          |
| `parch`      | Number of parents/children aboard          |
| `fare`       | Passenger fare                             |
| `class`      | Passenger class as a categorical feature   |
| `who`        | Passenger category                         |
| `adult_male` | Whether the passenger was an adult male    |
| `alone`      | Whether the passenger was travelling alone |

### Target

```text
survived
```

## 🛠️ Technologies & Libraries

* **Python**
* **Pandas** — data manipulation and analysis
* **NumPy** — numerical operations
* **Matplotlib** — data visualisation
* **Seaborn** — dataset and visualisation
* **Scikit-learn** — preprocessing, model training, hyperparameter tuning and evaluation
* **XGBoost** — advanced gradient boosting classification

## 🔄 Machine Learning Workflow

### 1. Data Preparation

The Titanic dataset is loaded using Seaborn and the selected features are separated from the target variable.

The dataset is then split into:

* **80% training data**
* **20% testing data**

Stratified sampling is used to maintain the target-class distribution between the training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 2. Data Preprocessing

Separate preprocessing pipelines are created for numerical and categorical features.

#### Numerical Features

Missing numerical values are replaced using the median, followed by standardisation using `StandardScaler`.

```text
Missing values → Median Imputation → Standard Scaling
```

#### Categorical Features

Missing categorical values are replaced using the most frequent value, followed by one-hot encoding.

```text
Missing values → Most Frequent Imputation → One-Hot Encoding
```

A `ColumnTransformer` combines both preprocessing pipelines.

### 3. Model Evaluation & ROC-AUC Analysis
The models are comprehensively evaluated using multiple metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC. Visualizations include classification reports, confusion matrices, and combined ROC curves to assess the true positive vs. false positive tradeoff.

### 4. Feature Importance & Interpretability
To interpret how models make their decisions, the project extracts and visualizes importance metrics:   
* **Tree-based models (Random Forest & XGBoost)**: Feature importance scores are extracted and plotted to identify the most significant transformed features.
* **Logistic Regression**: Coefficient magnitudes are extracted and plotted as a bar chart to analyze the relative impact (positive or negative) of each feature. 

## 📏 Model Evaluation

The models are evaluated using:

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Confusion matrices are generated to visualise:

* True positives
* True negatives
* False positives
* False negatives

### Test Accuracy

The final test-set accuracy is also calculated for each model.

> **Note:** Model results may vary depending on the environment and library versions.

## 📈 Visualisations

The notebook generates visualisations including:

1. Random Forest, Logistic Regression, Support Vector Machine, K-Nearest Neighbour, Gradient Boosting, and XGBoost confusion matrix
2. Random Forest feature importance
3. Logistic Regression coefficient magnitude
4. Gradient Boosting feature importance
5. XGBoost feature importance
6. Model accuracy comparison
7. Model F1 score comparison
8. ROC curves model comparison

These visualisations help evaluate model performance and understand which features influence predictions.

## 📁 Project Structure

```text
titanic-survival-prediction/
│
├── titanic_survival_prediction.ipynb
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn seaborn xgboost
```

### 3. Run the Notebook

Open the notebook using Jupyter:

```bash
jupyter notebook titanic_survival_prediction.ipynb
```

Alternatively, the notebook can be opened using **JupyterLab** or **Google Colab**.

## 💡 Key Concepts Demonstrated

This project demonstrates practical applications of several machine-learning concepts:

* Binary classification
* Train-test splitting
* Stratified sampling
* Missing-value imputation
* Feature scaling
* One-hot encoding
* Machine-learning pipelines
* ColumnTransformer
* Random Forest classification
* Logistic Regression
* Hyperparameter tuning
* GridSearchCV
* Stratified K-Fold cross-validation
* Evaluation metrics: Confusion matrices, Precision, Recall, F1, and ROC-AUC
* Model interpretability via Feature Importance and Coefficient Analysis   

## 🔍 Learning Outcomes

Through this project, I explored how to:

* Prepare real-world datasets for machine learning
* Handle both numerical and categorical data within a single pipeline
* Build reusable preprocessing and model pipelines
* Tune machine-learning models using cross-validation
* Compare different classification algorithms
* Evaluate classification performance using multiple metrics
* Interpret trained models through feature importance and coefficients
