# Loan Prediction Project

![Loan Status Prediction](https://github.com/user-attachments/assets/4bcc9df7-a846-4425-9958-efab8f209272)
</br>

![Loan Status Prediction](https://github.com/user-attachments/assets/d72ce6f7-910a-49c0-b973-39485b18a704)
</br>


## Overview
This project aims to automate the loan eligibility process in real-time based on customer details provided in an online application form. The details include information such as gender, marital status, education, number of dependents, income, loan amount, credit history, and more.

### Dataset
- **Source**: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- **Rows**: 615
- **Columns**: 13

The dataset includes variables that influence loan eligibility decisions, providing insights into customer segments eligible for loans.

## Problem Statement
The goal is to identify customer segments eligible for loans, enabling the company to specifically target them. This project involves:
- Data collection and preprocessing
- Exploratory data analysis and visualization
- Model training and evaluation
- Building a predictive system

![Problem Statement](https://github.com/user-attachments/assets/610c77c1-3091-42e9-9a5c-f643125f8341)
</br>


---

## Work Flow

![Work Flow](https://github.com/user-attachments/assets/4a7c6220-cb36-48c1-bc34-20ef92c9cb56)
</br>


---

## Installation
### Required Tools and Libraries
Install the necessary tools and libraries using:
```bash
pip install gradio
```

---

## Workflow
### 1. Data Collection and Preprocessing
#### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
#### Load Dataset
```python
# Load the dataset
file_path = "/root/.cache/kagglehub/datasets/ninzaami/loan-predication/versions/1/train_u6lujuX_CVtuZ9i (1).csv"
data = pd.read_csv(file_path)
```
#### Preprocessing Steps
- Handling missing values
- Encoding categorical data
- Splitting data into training and testing sets

---

### 2. Exploratory Data Analysis
#### Visualizations
- Education vs Loan Status
  
![Sample Data Visualization](https://github.com/user-attachments/assets/c0667b5c-4793-4870-b72c-fef6c2844215)
</br>

- Marital Status vs Loan Status
  
![Sample Data Visualization](https://github.com/user-attachments/assets/c685e6db-5663-427b-b185-2daab3626c69)
</br>

- Gender vs Loan Status
  
![Sample Data Visualization](https://github.com/user-attachments/assets/b2ba8347-3216-4e3b-9322-b2368e39d74d)
</br>

- Property Area vs Loan Status
  
![Sample Data Visualization](https://github.com/user-attachments/assets/dae8d828-f509-49cc-843c-dbd7a219e1ec)
</br>

- Self Employed vs Loan Status
  
![Sample Data Visualization](https://github.com/user-attachments/assets/be0cc433-8327-4011-b0e5-6e1be48ace89)
</br>


```python
sns.countplot(x='Education', hue='Loan_Status', data=loan_data)
```
---

### 3. Model Training and Evaluation
#### Model: Support Vector Machine (SVM)
```python
# Training the SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
```
#### Evaluation Metrics
- **Accuracy**: `0.80` on training data, `0.82` on testing data
- **Confusion Matrix**: Detailed view of predictions

---

### 4. Predictive System
#### Function for Prediction
```python
def predict_loan_status(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = classifier.predict(input_array)
    return "Loan Approved" if prediction[0] == 1 else "Loan Not Approved"
```
#### Example
```python
sample_data = [1, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2]
result = predict_loan_status(sample_data)
print("Prediction Result:", result)
```
---

### 5. Gradio Interface

![Gradio Interface Screenshot](https://github.com/user-attachments/assets/e5ae25fd-a6dc-400a-b4bc-eaa0ea81568f)
</br>

![Gradio Interface Screenshot](https://github.com/user-attachments/assets/56175f26-6104-4e42-a86c-3df6c6525c72)
</br>


#### Setup
```python
import gradio as gr

# Gradio Interface
interface = gr.Interface(
    fn=gradio_predict,
    inputs=[...],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Loan Approval Prediction",
    description="Enter the required details to predict if a loan will be approved."
)

interface.launch()
```
---

## Results
- **Training Accuracy**: 79.95%
- **Testing Accuracy**: 82.29%

## Future Improvements
- Implementing additional algorithms like Random Forest or XGBoost
- Enhancing data preprocessing techniques
- Developing a user-friendly frontend

---

## Acknowledgments
- [Kaggle Dataset by Ninzaami](https://www.kaggle.com/datasets/ninzaami/loan-predication)

## License
This project is licensed under the MIT License.
