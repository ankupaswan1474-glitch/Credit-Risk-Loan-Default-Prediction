# Credit Risk Loan Default Prediction

## Machine Learning & Explainable AI for Credit Risk

An end-to-end credit risk modelling project focused on predicting loan
default using supervised machine learning and explainable AI. The
project uses the **Home Equity (HMEQ) dataset** and compares Logistic
Regression, Random Forest, and XGBoost models, followed by
hyperparameter tuning and SHAP-based model interpretation.

> **Portfolio project:** The repository demonstrates the complete
> workflow from data cleaning and exploratory analysis to model
> development, evaluation, explainability, and business recommendations.

------------------------------------------------------------------------

## Project Overview

Financial institutions need to identify borrowers who are likely to
default before approving loans. A model that misses actual defaulters
can expose a lender to avoidable credit losses, while a model that flags
too many good borrowers can unnecessarily restrict credit.

The objective of this project is to build a binary classification model
that predicts whether a home equity loan will default:

-   `BAD = 1` → Default / severely delinquent
-   `BAD = 0` → Non-default / repaid

Because the dataset contains approximately 20% defaulters, **Recall is
treated as the primary evaluation metric**, with Precision, F1-Score,
ROC-AUC and PR-AUC used to provide a broader assessment.

------------------------------------------------------------------------

## Dataset

The project uses the **HMEQ (Home Equity)** dataset containing:

-   **5,960 loan records**
-   **13 original variables**
-   **1 binary target variable:** `BAD`
-   Approximately **80.05% non-defaults**
-   Approximately **19.95% defaults**

### Key Variables

  Variable    Description
  ----------- ------------------------------------
  `BAD`       Default indicator / target
  `LOAN`      Loan amount
  `MORTDUE`   Outstanding mortgage balance
  `VALUE`     Current property value
  `REASON`    Loan purpose
  `JOB`       Occupation category
  `YOJ`       Years at current employer
  `DEROG`     Number of major derogatory reports
  `DELINQ`    Number of delinquent credit lines
  `CLAGE`     Age of oldest credit line
  `NINQ`      Recent credit inquiries
  `CLNO`      Number of existing credit lines
  `DEBTINC`   Debt-to-income ratio

------------------------------------------------------------------------

## Project Workflow

The project follows an end-to-end credit risk modelling workflow:

``` text
Raw HMEQ Dataset
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Missing Value Treatment
       ↓
Outlier Treatment
       ↓
Categorical Encoding
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Baseline Models
       ↓
Hyperparameter Tuning
       ↓
Model Comparison
       ↓
Final Model Selection
       ↓
SHAP Explainability
       ↓
Business Recommendations
```

------------------------------------------------------------------------

## Data Preprocessing

### Missing Values

-   Numerical variables → **Median imputation**
-   Categorical variables → **Mode imputation**

Median imputation was preferred for numerical variables because several
financial variables are skewed and contain outliers.

### Outlier Treatment

IQR-based capping / Winsorization was applied to numerical variables.

The objective was not to remove potentially meaningful high-risk
observations, but to reduce the disproportionate influence of extreme
values during model training.

### Categorical Encoding

Categorical variables such as `REASON` and `JOB` were converted using
**One-Hot Encoding** with `drop_first=True`.

### Train-Test Split

-   **80% training**
-   **20% testing**
-   **Stratified sampling**
-   `random_state = 42`

The stratified split preserves the approximately 20% default rate in
both training and testing data.

### Feature Scaling

`StandardScaler` was used for standardized numerical inputs. Scaling was
fitted on the training data and then applied to the test data to avoid
data leakage.

------------------------------------------------------------------------

## Exploratory Data Analysis

The EDA examines relationships between borrower characteristics and
default behaviour.

Key risk patterns identified include:

-   **DEBTINC** is a major predictor of default risk.
-   Higher **DELINQ** is associated with increased default probability.
-   **DEROG** provides an important signal of previous credit
    deterioration.
-   **NINQ** can indicate recent credit-seeking behaviour and financial
    stress.
-   **VALUE** and collateral characteristics are relevant to borrower
    risk.
-   Employment stability (`YOJ`) and loan purpose also provide useful
    segmentation signals.

A correlation analysis also identified a strong relationship between
`MORTDUE` and `VALUE`, while `DEBTINC` showed relatively independent
predictive information.

------------------------------------------------------------------------

## Models

Three classification algorithms were evaluated.

### 1. Logistic Regression

Used as an interpretable baseline and traditional credit-scoring
benchmark.

### 2. Random Forest

Used to capture nonlinear relationships and feature interactions while
providing strong classification performance.

### 3. XGBoost

Used as the primary gradient-boosting model because of its ability to
model nonlinear relationships and complex interactions.

------------------------------------------------------------------------

## Model Evaluation

Because missing a true defaulter can be particularly costly in credit
risk, **Recall for the default class is the primary metric**.

The models were evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   ROC-AUC
-   PR-AUC
-   Confusion Matrix
-   ROC Curve
-   Precision-Recall Curve

### Baseline Performance

  -------------------------------------------------------------------------------
  Model          Accuracy   Precision     Recall         F1    ROC-AUC     PR-AUC
  ------------ ---------- ----------- ---------- ---------- ---------- ----------
  Logistic         80.45%      55.81%     10.08%     17.08%     67.85%     36.64%
  Regression                                                           

  Random           88.93%      76.50%     64.29%     69.86%     95.16%     83.02%
  Forest                                                               

  XGBoost          89.43%      80.11%     62.61%     70.28%     94.28%     83.70%
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## Hyperparameter Tuning

Hyperparameter optimization was performed using cross-validation with
**Recall as the primary scoring objective**.

### Logistic Regression

`GridSearchCV` was used to search across regularization and solver
configurations.

### Random Forest

`RandomizedSearchCV` was used across parameters including:

-   Number of estimators
-   Maximum depth
-   Minimum samples split
-   Minimum samples leaf
-   Maximum features

### XGBoost

`RandomizedSearchCV` was used across parameters including:

-   Number of estimators
-   Learning rate
-   Maximum depth
-   Subsample
-   Column sampling
-   Gamma

------------------------------------------------------------------------

## Final Model

### Tuned XGBoost

The tuned XGBoost model produced the strongest overall performance:

  Metric            Result
  ----------- ------------
  Accuracy      **90.94%**
  Precision     **84.21%**
  Recall        **67.23%**
  F1-Score      **74.77%**
  ROC-AUC       **95.26%**

The tuned model achieved the highest Recall, Precision, F1-Score and
ROC-AUC among the evaluated final models, making it the selected model
for this project.

### Why XGBoost?

-   Strongest overall discrimination
-   Highest ROC-AUC: **95.26%**
-   Highest Precision: **84.21%**
-   Highest Recall among the final tuned models: **67.23%**
-   Best F1-Score: **74.77%**

------------------------------------------------------------------------

## Threshold Analysis

The default classification threshold of `0.50` is not necessarily
optimal for credit risk.

An additional threshold analysis showed that lowering the XGBoost
threshold to approximately **0.35** can increase Recall to around
**73%**, with approximately **76% Precision**.

This illustrates an important credit-risk principle:

> The classification threshold should be selected based on the relative
> business cost of missed defaulters versus false alarms, rather than
> automatically using 0.50.

------------------------------------------------------------------------

## Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to make model
predictions more interpretable.

The analysis includes:

-   Global feature importance
-   SHAP bar plots
-   SHAP beeswarm plots
-   Individual waterfall explanations
-   Force plots
-   Feature dependence analysis

The explainability layer helps answer:

> **Why did the model classify this applicant as high risk?**

This is particularly important in credit risk because model outputs
should be interpretable and subject to human oversight.

------------------------------------------------------------------------

## Key Risk Drivers

The analysis identified several important risk factors:

1.  **DEBTINC** --- debt-to-income burden
2.  **CLAGE** --- length of credit history
3.  **VALUE** --- collateral/property value
4.  **DELINQ** --- delinquent credit lines
5.  **DEROG** --- major derogatory credit events
6.  **NINQ** --- recent credit inquiries

These variables provide useful signals for understanding borrower risk
and can support risk segmentation and underwriting decisions.

------------------------------------------------------------------------

## Business Recommendations

The project translates model results into practical credit-risk
recommendations:

### 1. Debt-to-Income Screening

Use `DEBTINC` as an important early-warning variable and consider manual
underwriting review for highly leveraged applicants.

### 2. Multi-Factor Risk Scoring

Combine the strongest risk indicators rather than relying on a single
variable.

### 3. Explainable Credit Decisions

Use SHAP explanations alongside model predictions to help credit
officers understand individual risk assessments.

### 4. Decision Threshold Optimization

Select a classification threshold based on the lender's cost of false
negatives versus false positives.

### 5. Risk-Based Pricing

Use predicted default probability as one input to risk-based pricing and
lending terms, subject to appropriate governance.

### 6. High-Risk Segment Monitoring

Pay particular attention to applicants combining high debt burden with
other indicators of credit deterioration.

### 7. Model Monitoring

Monitor model performance and potential concept drift as new loan
performance data becomes available.

### 8. Future Class-Imbalance Improvements

Future versions could investigate cost-sensitive learning, class
weighting and oversampling methods such as SMOTE.

------------------------------------------------------------------------

## Repository Contents

``` text
Credit-Risk-Loan-Default-Prediction/
│
├── Credit Risk Loan Default Prediction.ipynb
├── Credit Risk Loan Default prediction data.csv
├── Credit Risk Loan Default Prediction Project Guide.pdf
├── Credit_Risk_Loan_Default_Prediction_Summary.docx.pdf
└── README.md
```

### Main Files

**`Credit Risk Loan Default Prediction.ipynb`**\
Complete Python notebook containing data preparation, EDA, modelling,
tuning, evaluation and SHAP analysis.

**`Credit Risk Loan Default prediction data.csv`**\
Dataset used for the analysis.

**`Credit Risk Loan Default Prediction Project Guide.pdf`**\
Project workflow and implementation guide.

**`Credit_Risk_Loan_Default_Prediction_Summary.docx.pdf`**\
Detailed project report covering methodology, results and business
recommendations.

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost
-   SHAP
-   Jupyter Notebook

### Machine Learning

-   Logistic Regression
-   Random Forest
-   XGBoost
-   GridSearchCV
-   RandomizedSearchCV
-   Stratified train-test split

### Explainability

-   SHAP TreeExplainer
-   Feature importance
-   Beeswarm plots
-   Waterfall plots
-   Dependence plots

------------------------------------------------------------------------

## Key Takeaways

This project demonstrates a complete credit risk modelling lifecycle:

-   Data quality assessment
-   Exploratory credit-risk analysis
-   Missing-value and outlier treatment
-   Feature engineering and encoding
-   Supervised classification
-   Imbalanced-class evaluation
-   Hyperparameter optimization
-   Model comparison
-   Explainable AI
-   Business interpretation

The final tuned XGBoost model achieved **90.94% accuracy, 84.21%
precision, 67.23% recall, 74.77% F1-score and 95.26% ROC-AUC** on the
test set.

The project also demonstrates why model evaluation in credit risk should
go beyond accuracy and incorporate the cost of missed defaulters,
decision thresholds, interpretability and business context.

------------------------------------------------------------------------

## Disclaimer

This project is for **educational and portfolio purposes**. Model
performance on the HMEQ dataset should not be interpreted as evidence
that the model is suitable for direct deployment in a real lending
environment.

Real-world credit decisions require additional validation, calibration,
fairness testing, regulatory review, monitoring, governance and
institution-specific cost considerations.
