# Vehicle Dataset Analysis


![car Price Banner](https://github.com/user-attachments/assets/f637caf2-1f44-41ae-9feb-c5569844734f)
</br>

![Banner](https://github.com/user-attachments/assets/4bc4537f-e7d5-46d3-a914-5f41d6c9bb0a)


## Table of Contents

- [About the Project](#about-the-project)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results and Visualizations](#results-and-visualizations)
- [Conclusion](#conclusion)

---

## About the Project

This project focuses on analyzing a dataset of used vehicles to predict their selling prices using various machine learning models. The dataset includes attributes such as car name, year of manufacture, kilometers driven, fuel type, seller type, transmission, and owner details.

---

## Dataset Description

The dataset consists of the following columns:

- **name**: Name of the vehicle.
- **year**: Manufacturing year.
- **selling_price**: Price at which the vehicle is sold.
- **km_driven**: Kilometers driven by the vehicle.
- **fuel**: Fuel type (Petrol, Diesel, CNG).
- **seller_type**: Type of seller (Dealer, Individual).
- **transmission**: Transmission type (Manual, Automatic).
- **owner**: Ownership details.

**Example Shape of Dataset**: (301, 9)

---

## Work Flow

![Work Flow](https://github.com/user-attachments/assets/354c6555-5b53-4e7f-ac4f-9fffa3693599)


---


## Project Structure

```plaintext
├── dataset
│   └── car_data.csv
├── notebooks
│   └── data_analysis.ipynb
├── src
│   └── model_training.py
│   └── evaluation.py
├── results
│   └── visualizations
└── README.md
```

---

## Requirements

Install the necessary dependencies:

```bash
pip install --upgrade xgboost scikit-learn matplotlib seaborn pandas numpy gradio
```

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/AdMub/Data-Science-Project.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Car Price Prediction
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis script:
   ```bash
   python src/model_training.py
   ```

---

## Data Collection and Preprocessing

### Steps

1. **Load the Dataset**:
   ```python
   car_data = pd.read_csv('dataset/car_data.csv')
   ```

2. **Check for Missing Values**:
   ```python
   missing_values = car_data.isnull().sum()
   print(missing_values)
   ```

3. **Encode Categorical Variables**:
   ```python
   car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}}, inplace=True)
   ```

4. **Split Features and Target**:
   ```python
   X = car_data.drop(columns=['name', 'selling_price'], axis=1)
   y = car_data['selling_price']
   ```

---

## Model Training

### Algorithms Used:

1. **Linear Regression**
2. **Lasso Regression**
3. **Gradient Boosting Regressor**

#### Example (Linear Regression):

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

## Model Evaluation

### Metrics:

- **R-squared Error**
- **Mean Absolute Error (MAE)**

#### Example:

```python
from sklearn.metrics import r2_score, mean_absolute_error

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"R-squared: {r2}, MAE: {mae}")
```

---

## Results and Visualizations

### Scatter Plot (Training Data):

![Scatter Plot](https://github.com/user-attachments/assets/fbbac454-945d-42be-a9ca-34070723df79)
</br>

### Scatter Plot (Training Data):
![Scatter Plot](https://github.com/user-attachments/assets/2c5bb59a-1135-49e4-96b5-5382feacec9a)
</br>

---

## Conclusion

This project demonstrates the power of machine learning in predicting vehicle prices accurately. By leveraging multiple algorithms and tuning their hyperparameters, significant accuracy was achieved. Future work may include incorporating more features or external data sources for enhanced predictions.

---
