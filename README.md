# HR Employee Turnover Classifier — Logistic Regression & Regularization Modeling

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Data%20Visualization-388E3C?style=for-the-badge)](https://seaborn.pydata.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

An end-to-end **Human Resources Analytics & Machine Learning Classification** project designed to predict employee turnover using **Logistic Regression** and **Linear/Logistic Regularization techniques (L1 Lasso & L2 Ridge)**. 

By analyzing employee engagement metrics, compensation structures, workplace satisfaction, and interaction features, this model empowers People Operations and HR leaders to proactively identify attrition risks and make data-driven retention strategies.

---

##  Table of Contents
- [Executive Summary](#-executive-summary)
- [Dataset Overview & Data Dictionary](#-dataset-overview--data-dictionary)
- [Project Architecture & Repository Structure](#-project-architecture--repository-structure)
- [Methodology & ML Workflow](#-methodology--ml-workflow)
- [Model Evaluation & Performance Comparison](#-model-evaluation--performance-comparison)
- [Key Findings & Regularization Insights](#-key-findings--regularization-insights)
- [Getting Started & Installation](#-getting-started--installation)
- [Usage & Running the Notebook](#-usage--running-the-notebook)
- [Business Impact & HR Recommendations](#-business-impact--hr-recommendations)

---

##  Executive Summary

Employee attrition carries substantial direct and indirect financial costs for organizations—ranging from recruitment expenses to lost productivity and team morale disruption. 

In this repository, we develop a predictive binary classifier on **1,350 employee records** to classify whether an employee will stay (`0`) or leave (`1`). We explore baseline Logistic Regression and benchmark it against **L1 (Lasso)** and **L2 (Ridge)** regularized models to handle collinearity introduced by engineered polynomial interaction features.

###  Key Highlights
- **Best Performing Classifier**: **L1 Regularized (Lasso) Logistic Regression (`C=0.5`, `solver='liblinear'`)**
- **Overall Accuracy**: **87.04%** (+1.11% gain over unregularized baseline)
- **Turnover Class Precision**: **88.00%** (minimizing false-positive attrition flags)
- **Feature Selection**: L1 Regularization successfully suppresses noisy polynomial interaction terms while preserving core retention signals.

---

##  Dataset Overview & Data Dictionary

The project utilizes `employee_turnover.csv`, containing **1,350 rows** and **16 columns** (no missing values). Predictors include continuous normalized metrics, employee demographic attributes, compensation indices, and engineered interaction terms.

| Feature Name | Type | Description |
| :--- | :---: | :--- |
| `Job_Satisfaction` | Continuous (`float64`) | Employee's reported satisfaction score at the workplace (normalized `[0.0, 1.0]`). |
| `Performance_Rating` | Continuous (`float64`) | Recent performance evaluation score of the employee. |
| `Years_At_Company` | Continuous (`float64`) | Tenure of the employee within the organization. |
| `Work_Life_Balance` | Continuous (`float64`) | Perceived balance between professional demands and personal life. |
| `Distance_From_Home` | Continuous (`float64`) | Commute distance from home to the office. |
| `Monthly_Income` | Continuous (`float64`) | Monthly compensation level. |
| `Education_Level` | Continuous (`float64`) | Educational qualification attainment level. |
| `Age` | Continuous (`float64`) | Age of the employee. |
| `Num_Companies_Worked` | Continuous (`float64`) | Number of previous employers prior to joining the current company. |
| `Employee_Role` | Continuous (`float64`) | Encoded representation of the employee's role/level. |
| `Annual_Bonus` | Continuous (`float64`) | Annual bonus reward metric. |
| `Training_Hours` | Continuous (`float64`) | Total hours spent in professional training and skill development. |
| `Department` | Continuous (`float64`) | Encoded department identifier. |
| `Annual_Bonus_Squared` | Continuous (`float64`) | Engineered quadratic term ($Annual\_Bonus^2$) to capture non-linear bonus effects. |
| `Annual_Bonus_Training_Hours_Interaction` | Continuous (`float64`) | Interaction term ($Annual\_Bonus \times Training\_Hours$). |
| **`Employee_Turnover` (Target)** | **Binary (`int64`)** | **Turnover classification status (`0` = Stayed, `1` = Left / Attrited).** |

---

##  Project Architecture & Repository Structure

```text
logistic-regression-hr-classifier-model/
├── Logistic_Regression_Project.ipynb   # Complete interactive Jupyter Notebook (EDA, Modeling & Evaluation)
├── employee_turnover.csv               # Dataset containing 1,350 employee records
└── README.md                           # Comprehensive technical and business project documentation
```

---

##  Methodology & ML Workflow

1. **Exploratory Data Analysis & Data Validation**:
   - Confirmed dataset integrity (`1,350` records across `16` attributes, zero `NaN` values).
   - Inspected target variable distribution (`Employee_Turnover`) to ensure well-balanced classes for modeling.

2. **Data Partitioning**:
   - Stratified train-test split using `test_size=0.20` (`random_state=42`), allocating **1,080 samples** for model training and **270 samples** for unseen out-of-sample evaluation.

3. **Baseline Logistic Regression**:
   - Fit standard unregularized `LogisticRegression()` model on the training set to establish performance benchmarks.

4. **Linear Regression Regularization Study (`LassoCV`)**:
   - Investigated continuous L1 shrinkage across an array of penalty coefficients (`alphas = [0.001, 0.1, 1, 2, 5, 10, ...]`) and identified `alpha = 0.001` via 5-fold Cross-Validation (`LassoCV`) with $R^2 = 0.5500$ and $MSE = 0.1119$.

5. **Regularized Logistic Classification Modeling**:
   - **L1 Regularization (Lasso)**: Trained `LogisticRegression(penalty='l1', solver='liblinear', C=0.5)` to promote feature sparsity and eliminate collinear interaction noise.
   - **L2 Regularization (Ridge)**: Trained `LogisticRegression(penalty='l2', C=1, max_iter=200)` to shrink coefficient weights evenly.

---

##  Model Evaluation & Performance Comparison

All models were evaluated on the held-out test set (`n = 270`). Below is the side-by-side performance summary across all classification approaches:

| Model | Regularization | Hyperparameters | Overall Accuracy | Precision (Turnover = 1) | Recall (Turnover = 1) | F1-Score (Turnover = 1) | Macro Avg F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | None | Default | 85.93% | 0.87 | 0.82 | 0.84 | 0.86 |
| **L1 Regularized Logistic (Lasso)** | **L1** | **`C=0.5`, `liblinear`** | **87.04%** | **0.88** | **0.83** | **0.86** | **0.87** |
| **L2 Regularized Logistic (Ridge)** | L2 | `C=1.0` | 85.93% | 0.87 | 0.82 | 0.84 | 0.86 |

### Detailed Classification Report — L1 Regularized Classifier (Best Model)

```text
              precision    recall  f1-score   support

  0 (Stayed)       0.86      0.90      0.88       145
    1 (Left)       0.88      0.83      0.86       125

    accuracy                           0.87       270
   macro avg       0.87      0.87      0.87       270
weighted avg       0.87      0.87      0.87       270
```

---

##  Key Findings & Regularization Insights

1. **Why L1 Regularization Outperformed Unregularized & L2 Models**:
   - The dataset includes engineered features (`Annual_Bonus_Squared` and `Annual_Bonus_Training_Hours_Interaction`) that introduce multicollinearity with their parent variables (`Annual_Bonus` and `Training_Hours`).
   - Standard Logistic Regression and L2 Ridge retain all features, which can lead to variance inflation.
   - **L1 Lasso (`C=0.5`)** performs **automatic feature selection**, driving negligible or noisy coefficients to exactly zero. This reduces overfitting on interaction terms and boosts generalization accuracy on the test set from **85.93% to 87.04%**.

2. **High Precision on Turnover Detection**:
   - Achieving **88% precision** on `Employee_Turnover = 1` ensures that when HR teams flag an employee as high-risk for departure, the signal is highly reliable—preventing unnecessary retention interventions on employees likely to stay.

---

##  Getting Started & Installation

### Prerequisites
- Python **3.9+**
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/majisandhyatara-creator/logistic-regression-hr-classifier-model.git
cd logistic-regression-hr-classifier-model
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

---

##  Usage & Running the Notebook

To explore the entire data science pipeline, launch Jupyter Notebook from the project directory:

```bash
jupyter notebook Logistic_Regression_Project.ipynb
```

You can execute all cells sequentially (`Cell > Run All`) to:
1. Load and inspect `employee_turnover.csv`
2. Perform train/test splits
3. Recreate baseline classification results
4. Visualize Lasso regression regularization paths across varying alpha values
5. Verify the L1 and L2 regularized classifier evaluation metrics

---

##  Business Impact & HR Recommendations

- **Target Retention Interventions**: Prioritize proactive 1-on-1 check-ins and tailored career discussions for employees predicted as high-risk by the regularized model.
- **Holistic Compensation Strategy**: Because non-linear interaction terms (`Annual_Bonus_Squared` and training interaction) affect turnover probability, HR should pair compensation bonuses with meaningful growth opportunities (`Training_Hours`).
- **Regular Model Calibration**: Periodically retrain the classifier as workforce demographics and satisfaction surveys evolve over quarterly review cycles.

---

##  License & Contributing

Feel free to open issues or submit pull requests if you would like to contribute additional feature engineering, non-linear classifiers (e.g., Random Forests, XGBoost), or interactive dashboard integrations.
