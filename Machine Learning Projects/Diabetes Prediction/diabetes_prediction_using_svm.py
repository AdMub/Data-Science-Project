# -*- coding: utf-8 -*-
"""Diabetes Prediction using SVM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XRlLgWOSktzI74NAPdQ-EL06LQOfFbPv

## Installing require Tools
"""

!pip install gradio

"""## Importing the Dependencies"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import gradio as gr

"""## Data Collection and Analysis"""

# Loading the Diabetes dataset to a panda DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# Print the first 5 rows of the dataset
diabetes_dataset.head()

# Checking the number of Rows and Columns in this dataset
diabetes_dataset.shape

"""It implies we have 768 rows and 9 columns"""

# # Getting the statistical Description of the data
diabetes_dataset.describe()

# checking the features label in the column (Outcomes)
diabetes_dataset['Outcome'].value_counts()

"""0 --> Non-Diabetic

1 --> Diabetic
"""

# Group the diabetes dataset by the 'Outcome' column, which represents whether a person has diabetes (1) or not (0),
# and calculate the mean of all numerical columns for each group.
diabetes_dataset.groupby('Outcome').mean()

# Separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""## Data Standardization"""

# Create an instance of the StandardScaler class, which is used to standardize features
# by removing the mean and scaling to unit variance. This ensures that the data is
# normalized and ready for machine learning algorithms that are sensitive to feature scaling.
scaler = StandardScaler()

# Transform the data to ensure proper scaling.
scaler.fit(X)

# This scales the features to have a mean of 0 and a standard deviation of 1.
standardized_data = scaler.transform(X)

"""Or using another way"""

standardized_data = scaler.fit_transform(X)

print(standardized_data)

# Splitting Data into Dependent and Independent Features
X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""## Splitting Data into Training and Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

# Traing the model using Support Vector Machine
classifier = svm.SVC(kernel='linear')

# Training the SVM Classifier
classifier.fit(X_train, Y_train)

"""## Model Evaluation

Using Accuracy Score
"""

# Accuracy Score of the Training Data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy Score of the Test Data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

from sklearn.model_selection import GridSearchCV, cross_val_score

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(svm.SVC(), param_grid, scoring='accuracy', cv=5)
grid.fit(X_train, Y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

"""**GridSearchCV:** A tool for exhaustive search over specified hyperparameter values for an estimator.

**cross_val_score:** Performs cross-validation to evaluate a model's performance.


**param_grid** is a dictionary specifying the hyperparameters and their possible values to test:

  * **C:** Regularization parameter. Controls the trade-off between achieving a low error on the training data and minimizing model complexity. Lower C values encourage a simpler model, while higher C values prioritize fitting the training data better.

  * **kernel:** Specifies the kernel type to be used in the SVM. Options:

    * **'linear':** Linear kernel for linearly separable data.

    * **'rbf':** Radial Basis Function (RBF) kernel for non-linear data.

**svm.SVC():** The estimator (SVM classifier) we are optimizing.

**param_grid:** The dictionary of hyperparameters to search through.

**scoring='accuracy':** Evaluation metric used to score each combination of hyperparameters.


**cv=5:** The number of cross-validation splits. The data is split into 5 folds, and the model is trained on 4 folds and tested on the remaining 1 fold. This process is repeated 5 times (rotating the test fold), and the results are averaged.

The **fit** method trains the SVM model on the training data (**X_train** and **Y_train**) for every combination of hyperparameters specified in **param_grid**.

It evaluates each combination using 5-fold cross-validation.



The best combination of hyperparameters is **C=1** and **kernel='linear'**.

The highest average accuracy achieved during cross-validation is **78%**.

## Making a Predictive System and Using Gradio Interface
"""

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

input_data = (1,85,66,29,0,26.6,0.351,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

def gradio_chat_interface(input_data):
    try:
        # Convert the input string into a list of floats
        input_data = list(map(float, input_data.split(',')))

        # Change the input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_data_reshaped)

        # Make the prediction
        prediction = classifier.predict(std_data)

        # Prepare the output message
        if prediction[0] == 0:
            return "The person is NOT diabetic."
        else:
            return "The person IS diabetic."
    except Exception as e:
        return f"Error: {str(e)}. Please enter valid input in the correct format."

# Define the Gradio interface
interface = gr.Interface(
    fn=gradio_chat_interface,
    inputs="text",
    outputs="text",
    title="Diabetes Predictive System",
    description=(
        "This system predicts whether someone is diabetic or non-diabetic. "
        "Please enter 8 numerical values separated by commas (e.g., '5,166,72,19,175,25.8,0.587,51')."
    ),
)

# Launch the Gradio interface
interface.launch()

def predictive_system(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, classifier):
    # Collect the data into a list (the features)
    data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    # Convert the input data into a numpy array (suitable for prediction)
    input_data_as_numpy_array = np.asarray(data)

    # Reshape the data for prediction (model expects 2D array)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Use the classifier to predict the outcome (diabetes or not)
    prediction = classifier.predict(input_data_reshaped)
    print(prediction)

    # Check the prediction and print the result
    if prediction[0] == 0:
        print('The person is not diabetic')
    else:
        print('The person is diabetic')


# Test with sample input (these values are just an example)
predictive_system(6, 148, 72, 35, 0, 33.6, 0.627, 50, classifier)

# Function to handle the prediction logic
def predictive_system(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, classifier):
    # Collect the data into a list (the features)
    data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    # Convert the input data into a numpy array (suitable for prediction)
    input_data_as_numpy_array = np.asarray(data)

    # Reshape the data for prediction (model expects 2D array)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the data
    std_data = scaler.transform(input_data_reshaped)

    # Use the classifier to predict the outcome (diabetes or not)
    prediction = classifier.predict(std_data)

    # Check the prediction and return the result
    if prediction[0] == 0:
        return "The person is NOT diabetic."
    else:
        return "The person IS diabetic."

# Gradio interface function
def gradio_chat_interface(input_data):
    try:
        # Convert the input string into a list of floats
        input_data = list(map(float, input_data.split(',')))

        # Call the predictive_system function with the input data
        result = predictive_system(*input_data, classifier=classifier)

        return result
    except Exception as e:
        return f"Error: {str(e)}. Please enter valid input in the correct format."

# Define the Gradio interface
interface = gr.Interface(
    fn=gradio_chat_interface,
    inputs=gr.Textbox(label="Enter 8 Features", placeholder="Enter 8 comma-separated values (e.g., '5,166,72,19,175,25.8,0.587,51')", lines=1),
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Predictive System",
    description=(
        "This system predicts whether someone is diabetic or non-diabetic. "
        "Please enter 8 numerical values separated by commas, representing the following features:\n"
        "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age."
    ),
    theme="compact",  # Use the compact theme for a cleaner layout
    allow_flagging="never",  # Disable flagging
)

# Add custom CSS for a beautiful color scheme
css = """
#title {
    color: #4CAF50;  /* Green title */
    font-size: 30px;
    font-weight: bold;
}

#description {
    font-size: 16px;
    color: #555555;
}

#input_component {
    background-color: #e8f5e9; /* Light green background */
    border: 2px solid #4CAF50;
    font-size: 16px;
}

#output_component {
    background-color: #f1f8e9; /* Light green output area */
    border: 2px solid #4CAF50;
    font-size: 18px;
}

button {
    background-color: #4CAF50;  /* Green button */
    color: white;
    font-size: 18px;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}

button:hover {
    background-color: #45a049;  /* Darker green on hover */
}
"""

# Apply the custom CSS
interface.css = css

# Launch the Gradio interface
interface.launch()