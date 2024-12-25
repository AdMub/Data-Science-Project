# Diabetes Predictive System

![DALL·E 2024-12-25 12 44 07 - A modern, professional graphic of a diabetes prediction system interface  The image should include a user entering data into a simple,](https://github.com/user-attachments/assets/113fde1c-582b-49e3-98a2-98625429d60e)
</br>

![Annotation 2024-12-25 073703](https://github.com/user-attachments/assets/535e824a-375e-426b-9a37-d66774ff5df6)
</br>

## Overview

This project is a machine learning-based predictive system designed to determine whether a person is diabetic or not, based on certain health features. The model uses a dataset containing medical details of individuals and employs Support Vector Machine (SVM) as the classifier. The system has been integrated with a user-friendly Gradio interface that allows real-time predictions based on the input features.

## Features

* **Predictive Model:** The system uses a trained SVM classifier to predict diabetes status (diabetic or non-diabetic) based on health features.
* **Interactive Interface:** Users can input 8 numerical health features to receive a prediction (diabetic or non-diabetic).
* **Hyperparameter Tuning:** The system includes hyperparameter tuning using GridSearchCV to optimize the model for better accuracy.
* **Data Standardization:** The input data is standardized to ensure all features are on the same scale, improving the model’s performance.

## Project Work Flow

![Annotation 2024-12-25 073737](https://github.com/user-attachments/assets/e9a7ba21-bb33-4301-98f5-85d28e876301)
</br>

## Dataset

The model is trained on the Diabetes dataset, which consists of the following columns:

1. **Pregnancies:** Number of times pregnant.
2. **Glucose:** Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
3. **BloodPressure:** Diastolic blood pressure (mm Hg).
4. **SkinThickness:** Triceps skinfold thickness (mm).
5. **Insulin:** 2-Hour serum insulin (mu U/ml).
6. **BMI:** Body Mass Index (weight in kg / height in m²).
7. **DiabetesPedigreeFunction:** A function that provides the likelihood of diabetes based on family history.
8. **Age:** Age of the person.
   
The target variable (label) is **Outcome**, where:

* **0** indicates **Non-Diabetic**,
* **1** indicates **Diabetic**.
  
## Requirements
* Python 3.x
* Libraries:
  * pandas
  * numpy
  * scikit-learn
  * gradio

Install the necessary libraries with the following command:
* bash

  *pip install pandas numpy scikit-learn gradio*

## Model Building

### Data Preprocessing

1. **Loading the Data:** The dataset is loaded into a Pandas DataFrame.
2. **Data Standardization:** The features are standardized using StandardScaler to scale them to a mean of 0 and standard deviation of 1.
3. **Train-Test Split:** The data is split into training and test sets (80% training, 20% test).
4. **Model Training:** An SVM classifier with a linear kernel is trained on the dataset.
5. **Model Evaluation:** The model is evaluated using accuracy scores for both the training and test sets.
6. **Hyperparameter Tuning:** *GridSearchCV* is used to find the best hyperparameters for the model.

## Model Evaluation
**Accuracy:** The model achieved an accuracy of **78.3%** on the training data and **77.9%** on the test data.
**Best Parameters:** *{'C': 1, 'kernel': 'linear'}* with a cross-validation score of 78.0%.

## Gradio Interface

The Gradio interface allows users to input 8 numerical values (features) and receive a prediction indicating whether the person is diabetic or not.

### Input Format

* The user needs to provide the following 8 features in comma-separated format *(e.g., '5,166,72,19,175,25.8,0.587,51')*:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

### Prediction Logic

* Once the user inputs the data, the system preprocesses the input, standardizes it, and then feeds it to the trained SVM model.

* The prediction is based on the classifier’s result:
  * *0*: **Non-Diabetic**
  * *1*: **Diabetic**

** How to Use

1. **Launch the Gradio Interface:** After running the script, the Gradio interface will launch automatically in your web browser.

2. **Input the Data:** Enter 8 comma-separated values representing the features into the input field.

For example:
*5,166,72,19,175,25.8,0.587,51*

3. **Receive the Prediction:** After entering the data, the system will display whether the person is diabetic or not based on the model’s prediction.

## Example

### Example 1:

**Input:** *5,166,72,19,175,25.8,0.587,51*

**Output:** *The person is diabetic.*

### Example 2:

**Input:** *1,85,66,29,0,26.6,0.351,31*

**Output:** *The person is NOT diabetic.*

## Project Interface

Below are visuals showcasing the interface of the **Diabetes Predictive System**, highlighting its user-friendly design and functionality:

**1. Input Section**
The interface allows users to input eight comma-separated numerical values corresponding to the medical features required for the prediction.

![image](https://github.com/user-attachments/assets/28ca3823-75d1-4e8d-82d9-83b1b374110d)
</br>

![image](https://github.com/user-attachments/assets/5e20084d-3ac8-46d2-8f6a-8ddb6d11a586)
</br>

**2. Prediction Output** 

After input submission, the system processes the data and provides an accurate prediction indicating whether the individual is diabetic or not.

![image](https://github.com/user-attachments/assets/b93add4c-4956-4c5e-8d17-77da03c24531)
</br>

![image](https://github.com/user-attachments/assets/d9c68598-4fde-4b99-a139-8291386d8dce)
</br>

![image](https://github.com/user-attachments/assets/d1bd6d3d-315c-4592-9af3-089a24d3ae3b)
</br>

![image](https://github.com/user-attachments/assets/389e7d71-bac9-4fba-afae-e88b421578dc)
</br>


**3. Error Handling**

The interface is designed to handle errors gracefully, ensuring users receive meaningful feedback if inputs are invalid or improperly formatted.

![image](https://github.com/user-attachments/assets/11459644-1f11-4a9e-9b1a-7eb7816d58af)
</br>

![image](https://github.com/user-attachments/assets/5d9c4582-a162-4a75-a754-a40206a63e81)
</br>

## Conclusion

This project demonstrates the use of machine learning for predicting diabetes and showcases how such models can be deployed in real-time using Gradio to create interactive interfaces. With accurate predictions and a user-friendly interface, this system can help individuals assess their likelihood of having diabetes based on health metrics.

## Future Improvements

* **Expand the Model:** Integrate other machine learning algorithms such as Random Forests, Logistic Regression, etc.
* **Web Deployment:** Host the Gradio app on a server for wider access.
* **User Input Validation:** Add more robust error handling and input validation.

## License

This project is licensed under the MIT License.

🌟💡 Empowering Health Decisions Through AI! 🤖❤️
