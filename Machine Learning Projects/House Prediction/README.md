# California Housing Price Prediction

Welcome to the **California Housing Price Prediction** project! This project demonstrates the use of machine learning models to predict housing prices based on a dataset of housing attributes. It uses various Python libraries for data analysis, visualization, and modeling, showcasing different techniques and models.

![Housing Prices Banner](https://github.com/user-attachments/assets/34ea64c6-41a2-47fa-b0f3-cba41f2a0bf4)
</br>

![Housing Prices Banner](https://github.com/user-attachments/assets/2d5f70e3-e7d8-4f22-95c9-a40560ba418e)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Work Flow](#work-flow)
4. [Setup and Installation](#setup-and-installation)
5. [Data Preprocessing](#data-preprocessing)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Building](#model-building)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Performance Evaluation](#performance-evaluation)
10. [Comparison of Algorithms](#comparison-of-algorithms)
11. [Conclusion](#conclusion)
12. [Acknowledgements](#acknowledgements)

---

## Introduction

The **California Housing Price Prediction** project utilizes the `California Housing Dataset` to predict housing prices. This project demonstrates the full lifecycle of machine learning modeling, from data preprocessing to model evaluation and tuning.

---

## Features

- Exploratory Data Analysis (EDA) with visualizations
- Correlation analysis and heatmaps
- Regression modeling using advanced algorithms such as **XGBoost** and **Gradient Boosting**
- Hyperparameter tuning with `GridSearchCV`
- Performance comparison across multiple algorithms

---

## Work Flow

![Work Flow Diagram](https://github.com/user-attachments/assets/20411ff0-9b9f-4b11-a3b2-9eded66f795d)

---

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn`

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/AdMub/Data-Science-Project.git

# Navigate to the project directory
cd california-housing-prediction

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preprocessing

- The dataset is loaded using `fetch_california_housing`.
- Missing values are checked and handled.
- Features (`X`) and target (`y`) are separated for training and testing.
- Data transformations include scaling and polynomial feature generation.

---

## Exploratory Data Analysis (EDA)

### Correlation Heatmap

![Correlation Heatmap](https://github.com/user-attachments/assets/344d57ec-bf45-4157-9835-32c4db524996)


Key insights from EDA:
- Relationships between features and the target variable
- Visualization of feature distributions and correlations

---

## Model Building

### Algorithms Used:

1. **XGBoost**: Initial model with promising results.
2. **Gradient Boosting Regressor**: Another powerful ensemble method.
3. Additional models for comparison:
   - Random Forest
   - Ridge Regression
   - Lasso Regression
   - Support Vector Regressor

---

## Hyperparameter Tuning

- **GridSearchCV** was used to optimize hyperparameters for XGBoost.
- Best parameters obtained:

```python
Best parameters: {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 7,
    'n_estimators': 300,
    'subsample': 0.8
}
```

---

## Performance Evaluation

### Metrics:
- R-squared (R²)
- Mean Absolute Error (MAE)

| Model                  | R² (Test) | MAE (Test) |
|------------------------|-----------|------------|
| XGBoost               | 0.8345    | 0.3090     |
| Gradient Boosting     | 0.7769    | 0.3717     |
| Random Forest         | 0.8016    | 0.3312     |
| Ridge Regression      | 0.5930    | 0.5351     |
| Lasso Regression      | 0.2889    | 0.7659     |
| Support Vector Regressor | -0.0108 | 0.8613     |

### Visualizations

#### Training Data: Actual vs Predicted Prices

![Training Data Scatter](https://github.com/user-attachments/assets/94b0fbae-1c40-4cb3-86a4-50645a62169c)

#### Test Data: Actual vs Predicted Prices

![Test Data Scatter](https://github.com/user-attachments/assets/7a8e2916-da82-46d9-9d1c-99330f310b46)


---

## Comparison of Algorithms

The project explored multiple models, highlighting the effectiveness of XGBoost with tuned hyperparameters for housing price prediction.

---

## Conclusion

The project successfully demonstrates:
- The importance of preprocessing and EDA.
- Building robust models with hyperparameter optimization.
- Visualizations and metrics to evaluate and compare models.

---

## Acknowledgements

Special thanks to the creators of the **California Housing Dataset** and the open-source contributors for the Python libraries used in this project.

Feel free to contribute to the project or reach out with suggestions!
