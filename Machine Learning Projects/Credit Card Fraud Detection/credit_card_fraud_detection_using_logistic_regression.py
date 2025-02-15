# -*- coding: utf-8 -*-
"""Credit Card Fraud Detection using Logistic Regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NAmkewgXpOvE8KZe8QR0gina67gD8zzY

#**Credit Card Fraud Detection**

**Anonymized credit card transactions labeled as fraudulent or genuine**

## **About Dataset**

### **Context**

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

### **Content**

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

### **Update (03/05/2021)**

A simulator for transaction data has been released as part of the practical handbook on Machine Learning for Credit Card Fraud Detection - https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html. We invite all practitioners interested in fraud detection datasets to also check out this data simulator, and the methodologies for credit card fraud detection presented in the book.

### **Acknowledgements**

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project

**Please cite the following works:**

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

Yann-Aël Le Borgne, Gianluca Bontempi Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook

Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé, Gianluca Bontempi Incremental learning strategies for credit cards fraud detection, IInternational Journal of Data Science and Analytics

# **Installing require Tools**
"""

pip install gradio

"""#**Import the Libraries**"""

## import some basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gradio as gr

"""#**Data Collecction and Preprocessing**"""

# Loading the dataset into a pandas DataFrame
creditcard_data = pd.read_csv('creditcard.csv')  # Use this to identify issues

# Display the first few rows
print("First 5 rows of the dataset:")
creditcard_data.head()

# Display the last few rows
print("Last 5 rows of the dataset:")
creditcard_data.tail()

# Checking the shape of the dataset
creditcard_data.shape    # Outputs the number of rows and columns in the dataset.

# Statistical description of the dataset
print("\nStatistical Description of the Dataset:")
creditcard_data.describe()

# Check for missing values
missing_values = creditcard_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Geerating some Information about the dataset
creditcard_data.info()

# Distribution of Legit transactions & Fraudulent Transactions
creditcard_data['Class'].value_counts()

"""## This Dataset is highly Unbalanced

**0 --->  Normal Transaction**

**1 ---> Fraudulent Transaction**

"""

# Separating the data for analysis
legit = creditcard_data[creditcard_data.Class == 0]
fraud = creditcard_data[creditcard_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# Statistical Measures of the data
legit.Amount.describe()

fraud.Amount.describe()

# Compare the values for both transaction
creditcard_data.groupby('Class').mean()

"""## **Under-Sampling**


Build a sample dataset containing similar distribution of Legit Transactions and Fraudulent Transactions



**Number of Normal Transactions --->53417**

**Number of Fraudulent Transactions ---> 153**



Picking randomly 153-data from the Legit dataset to equal to the number of Fraud dataset


**NB:**

But this method isn't the method for this data because amount have been lost
"""

legit_sample = legit.sample(n=153)

# Concatenating the two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Display the first few rows
print("First 5 rows of the dataset:")
new_dataset.head()

# Checking of Legit transactions & Fraudulent Transactions
new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

"""# **Splitting the data into Features & Target**

"""

X = new_dataset.drop(columns='Class', axis=1)
y= new_dataset['Class']

print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

"""# **Splitting the dataset into Training and Test sets**"""

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=3)
print("\nDataset split completed:")
print(f"Total samples: {X.shape[0]}, Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# checking the number of Test and Train dataset
print(X.shape, X_train.shape, X_test.shape)

"""# **Model Training**"""

# Initialize the model
logReg_model = LogisticRegression()

# Training the LogisticRegression model with train data
logReg_model.fit(X_train, y_train)

"""#**Model Evaluation**"""

# Calculate accuracy on the training data
X_train_pred = logReg_model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_pred, y_train)
print('Accuracy on training data : ', train_data_accuracy)

# Generate and display the confusion matrix on the training data
# The confusion matrix shows the counts of True Positives, True Negatives, False Positives, and False Negatives
conf_matrix = confusion_matrix(y_train, X_train_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Generate and display the classification report
# The classification report includes precision, recall, F1-score, and support for each class
class_report = classification_report(y_train, X_train_pred, target_names=["Normal Transaction (0)", "Fraudulent Transaction (1)"])
print("\nClassification Report:")
print(class_report)

# accuracy on the test data
X_test_pred = logReg_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, y_test)
print('Accuracy on test data : ', test_data_accuracy)

# Generate and display the confusion matrix on the test data
# The confusion matrix shows the counts of True Positives, True Negatives, False Positives, and False Negatives
conf_matrix = confusion_matrix(y_test, X_test_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Generate and display the classification report
# The classification report includes precision, recall, F1-score, and support for each class
class_report = classification_report(y_test, X_test_pred, target_names=["Normal Transaction (0)", "Fraudulent Transaction (1)"])
print("\nClassification Report:")
print(class_report)