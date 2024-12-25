# Sonar Mines vs. Rocks Classification Project

## Project Overview

This project focuses on building a machine learning model to classify sonar signals bounced off metal cylinders (mines) and rocks. Using the Sonar dataset, the project implements data analysis, preprocessing, model training, and evaluation techniques to create an effective classification system. The project employs logistic regression as the core classification algorithm.

## Dataset

**Name:** Sonar, Mines vs. Rocks

**Description:** The dataset contains 208 samples, each with 60 features representing the energy within specific frequency bands of sonar signals. Each sample is labeled as either R (Rock) or M (Mine).

**Source:** The dataset was contributed by Terry Sejnowski in collaboration with R. Paul Gorman.

## Libraries Used

The following Python libraries were utilized:

**numpy**: For numerical computations and array manipulations.

**pandas:** For data manipulation and analysis.

**matplotlib.pyplot:** For data visualization.

**seaborn:** For advanced visualizations.

**sklearn.model_selection.train_test_split:** To split data into training and test sets.

**sklearn.linear_model.LogisticRegression:** For implementing logistic regression.

**sklearn.metrics:** For evaluating model performance (accuracy, confusion matrix, classification report).

## References

1. Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets." NeuProject Workflow

  ### 1. Data Collection and Preprocessing

  #### Data Loading:

The dataset is loaded into a pandas DataFrame without headers.

Statistical summaries and value counts of the target column (**M** and **R**) are analyzed.

#### Feature and Target Splitting:

Features (**X**) and target (**y**) are extracted using various indexing techniques.

Equality checks were performed to confirm consistency across methods.

#### Dataset Shape:

The dataset comprises 208 rows and 61 columns (60 features and 1 target).

### 2. Data Splitting

The dataset is split into training and testing sets (90% training, 10% testing) while maintaining class distribution using stratification.

#### Parameters:

**test_size=0.1**

**stratify=y**

**random_state=1**

### 3. Model Training

#### Algorithm: Logistic Regression

##### Training:

The logistic regression model is trained using the training dataset.

Accuracy on both training and testing datasets is computed.

### 4. Model Evaluation

#### Metrics Used:

**Accuracy Score:** Measures the percentage of correctly classified samples.

**Confusion Matrix:** Evaluates true positive, true negative, false positive, and false negative rates.

**Classification Report:** Provides precision, recall, F1-score, and support metrics.

### 5. Predictive System

A predictive system is implemented to classify new sonar samples as either Rock or Mine based on the trained model.

**Sample Input:** A 60-dimensional array representing sonar signal frequencies.

**Prediction:** Outputs whether the sample is a Rock or Mine.

## Results

**Training Accuracy:** The model achieved a high accuracy on the training dataset.

**Test Accuracy:** The model demonstrated competitive performance on the test dataset.

## Methodology

Random train-test splitting was performed for independent aspect-angle experiments.

Data was clustered based on Euclidean distance for balanced aspect-angle experiments.

Logistic regression models with different configurations were compared against other classifiers and human performance.

neuraral Networks, Vol. 1.


## Getting Started

#### Prerequisites

Ensure the following Python libraries are installed:

pip install **numpy**, **pandas**, **matplotlib**, **seaborn**, **scikit-learn**

#### Running the Code

Clone the repository or download the project files.

Place the **sonar.csv** dataset in the same directory as the code.

Execute the script using:

python sonar_classification.py

#### Expected Output

Accuracy scores for both training and testing datasets.

Predictions for a sample input.

Visualizations of data distributions and confusion matrix.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

Special thanks to Terry Sejnowski and R. Paul Gorman for providing the dataset and methodology details.
