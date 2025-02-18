# **Breast Cancer Prediction Using Logistic Regression**

![Breast Cancer Prediction Using Logistic Regression](https://github.com/user-attachments/assets/2a825e45-9679-4562-ae2d-342f18a24859)

![Breast Cancer](https://github.com/user-attachments/assets/5307594a-22f9-4bfd-9519-499f9df97cf3)


## 📌 **Project Overview**
This project aims to develop a machine learning model to classify breast cancer cases as **Malignant** or **Benign** based on digitized features extracted from fine needle aspirate (FNA) of a breast mass. The dataset is sourced from the **UCI Machine Learning Repository**.

![Problem Statement](https://github.com/user-attachments/assets/7561052c-9613-4330-a3dd-405d0f82d3b5)


## ** Work Flow**

![Work Flow](https://github.com/user-attachments/assets/05bc6d3b-a6ea-4e76-b14c-ef528a8aa29e)


## 📂 **Dataset Overview**
The dataset contains **569 instances and 30 features**, describing characteristics of cell nuclei extracted from biopsy images.

### **Key Features**
- **Diagnosis**: Malignant (1) or Benign (0) (Target Variable)
- **Mean, Standard Error, and Worst values** for:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness
  - Compactness
  - Concavity
  - Concave Points
  - Symmetry
  - Fractal Dimension

### **Class Distribution**
- **357** Benign cases
- **212** Malignant cases

## 🚀 **Installation & Setup**
To set up the project environment, install the necessary dependencies:
```sh
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🔍 **Exploratory Data Analysis (EDA)**
### **Handling Missing Values**
The dataset has **no missing values**.

### **Data Visualizations**
#### **Distribution of Diagnosis**
```python
sns.countplot(x=df['label'])
plt.title("Diagnosis Distribution")
plt.show()
```
#### **Correlation Heatmap**
```python
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```

## 📊 **Feature Selection & Splitting**
```python
X = df.drop(columns='label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
```

## 🏗 **Model Training**
We use **Logistic Regression**, a simple yet effective model for binary classification.
```python
from sklearn.linear_model import LogisticRegression
logReg_model = LogisticRegression()
logReg_model.fit(X_train, y_train)
```

## 📉 **Model Evaluation**
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_acc = accuracy_score(y_train, logReg_model.predict(X_train))
test_acc = accuracy_score(y_test, logReg_model.predict(X_test))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```
**Results:**
- **Training Accuracy**: 95.07%
- **Test Accuracy**: 92.30%

## 🔎 **Making Predictions**
```python
input_data = [13.08,15.71,85.63,520,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183]
input_array = np.array(input_data).reshape(1, -1)
prediction = logReg_model.predict(input_array)
print("Malignant" if prediction[0] == 0 else "Benign")
```

## 🖼 **Project Image**
![Breast Cancer Prediction](image_link_here)

## 📌 **Conclusion**
- The **Logistic Regression model** provides a **92.30% accuracy** on test data.
- **Further improvements** could include feature engineering and hyperparameter tuning.

## 💡 **Future Enhancements**
- Experiment with **other models** like SVM, Random Forest, and XGBoost.
- Implement **Feature Selection** to remove redundant features.
- Deploy as a **web application** for easy accessibility.

---
📩 Feel free to connect for discussions or improvements!
