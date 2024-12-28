# California Housing Price Prediction

Welcome to the **California Housing Price Prediction** project! This project demonstrates the use of machine learning models to predict housing prices based on a dataset of housing attributes. It uses various Python libraries for data analysis, visualization, and modeling, showcasing different techniques and models.

![Housing Prices Banner](https://via.placeholder.com/900x300.png?text=California+Housing+Price+Prediction+Project)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Model Building](#model-building)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Performance Evaluation](#performance-evaluation)
9. [Comparison of Algorithms](#comparison-of-algorithms)
10. [Conclusion](#conclusion)
11. [Acknowledgements](#acknowledgements)

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

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn`

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/california-housing-prediction.git

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

![Correlation Heatmap](https://via.placeholder.com/600x400.png?text=Correlation+Heatmap)

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

![Training Data Scatter](https://via.placeholder.com/600x400.png?text=Train+Actual+vs+Predicted)

#### Test Data: Actual vs Predicted Prices

![Test Data Scatter](https://via.placeholder.com/600x400.png?text=Test+Actual+vs+Predicted)

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
