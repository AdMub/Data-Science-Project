# **Big Mart Sales Prediction**

![Big Mart Sales Prediction](https://github.com/user-attachments/assets/b4108e8e-2c5f-487b-973f-15acc0a4314b)



![Big Mart Sales Prediction](https://github.com/user-attachments/assets/2228e4b5-7703-4e18-b72b-95eee8b1b903)


## 📌 **Project Overview**
This project aims to develop a machine learning model to predict sales for different products across various Big Mart outlets. The dataset includes information on product attributes, store attributes, and sales data.

![Problem Statement](https://github.com/user-attachments/assets/263440b3-adbf-4b8f-873e-52aca1afce3a)

## 🚀 **Installation & Setup**
To set up the project environment, install the necessary dependencies:
```sh
pip install scikit-learn==1.2.2 xgboost pandas numpy matplotlib seaborn
```
## **Work Flow**

![Work Flow](https://github.com/user-attachments/assets/8333d97f-e576-4c89-99f7-fce24fb9f3ec)


## 📂 **Dataset Overview**
The dataset contains **8,523 rows and 12 columns**, with attributes related to product and outlet details:
- `Item_Identifier` – Unique ID for each product
- `Item_Weight` – Weight of the product
- `Item_Fat_Content` – Whether the product is low fat or regular
- `Item_Visibility` – The percentage of total display area allocated to the product
- `Item_Type` – Category of the product
- `Outlet_Identifier` – Store ID
- `Outlet_Establishment_Year` – Year the outlet was established
- `Outlet_Size` – Size of the outlet (Small, Medium, Large)
- `Outlet_Location_Type` – The type of city where the store is located
- `Outlet_Type` – The category of the outlet
- `Item_Outlet_Sales` – Sales of the product at the outlet (Target variable)

## 🔍 **Exploratory Data Analysis (EDA)**
### **Handling Missing Values**
- `Item_Weight` missing values are replaced with the column mean.
- `Outlet_Size` missing values are replaced using mode values based on `Outlet_Type`.

### **Data Visualizations**
#### **Distribution of Numeric Features**
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.figure(figsize=(6,6))
sns.histplot(big_mart_sales_data["Item_Weight"], kde=True)
plt.title("Item Weight Distribution")
plt.show()
```
#### **Correlation Heatmap**
```python
plt.figure(figsize=(6, 6))
sns.heatmap(big_mart_sales_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```

## 🔧 **Data Preprocessing**
- **Standardizing Categorical Features**:
  ```python
  big_mart_sales_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
  ```
- **Encoding Categorical Variables**:
  ```python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  big_mart_sales_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_sales_data['Item_Fat_Content'])
  ```

## 📊 **Feature Selection & Splitting**
```python
from sklearn.model_selection import train_test_split
X = big_mart_sales_data.drop(columns='Item_Outlet_Sales', axis=1)
y = big_mart_sales_data['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)
```

## 🏗 **Model Training**
We use **XGBoost Regressor**, a powerful boosting algorithm for regression tasks.
```python
from xgboost import XGBRegressor
xgb_model = XGBRegressor(random_state=1)
xgb_model.fit(X_train, y_train)
```

## 📉 **Model Evaluation**
```python
from sklearn import metrics
training_data_prediction = xgb_model.predict(X_train)
r2_train = metrics.r2_score(y_train, training_data_prediction)
mae_train = metrics.mean_absolute_error(y_train, training_data_prediction)
print(f"Training R-squared: {r2_train:.4f}, MAE: {mae_train:.4f}")
```
**Results:**
- **Training R-squared**: 0.8675
- **Test R-squared**: 0.5394
- **Mean Absolute Error (Test Data)**: 815.42

## 🖼 **Project Image**
![Data Analysis](image_link_here)

## 📌 **Conclusion**
- The XGBoost model provides a reasonable prediction with an **R-squared of 0.54** on test data.
- Further improvements could include **hyperparameter tuning** and **feature engineering**.

## 💡 **Future Improvements**
- Use **GridSearchCV** for hyperparameter optimization.
- Experiment with **other models** like Random Forest and Gradient Boosting.
- Perform **feature engineering** to improve model performance.

---
📩 **Feel free to connect for discussions or improvements!**
