# -*- coding: utf-8 -*-
"""Tensile Strength using ANN .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12syi5MVawN03nS43lr1hKSngYTPPWQOn

## **Data Analysis & Visualization**
"""

# Commented out IPython magic to ensure Python compatibility.
# Import necessary Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Load your data
tens_df = pd.read_excel('Compressive_Tensile_Data.xlsx', sheet_name='Tensile Strength')

# Display the first few rows
print("First 5 rows of the dataset:")
tens_df.head()

# Display the last few rows
print("Last 5 rows of the dataset:")
tens_df.tail()

# Checking the shape of the dataset
tens_df.shape    # Outputs the number of rows and columns in the dataset.

# Generating some Information about the dataset
tens_df.info()

# Statistical description of the dataset
print("\nStatistical Description of the Dataset:")
tens_df.describe()

# Check for missing values
missing_values = tens_df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Reorder of the column
new_order = [
    "Sample Type",
    "Tensile_Load_7D(KN)",
    "Tensile_Load_14D(KN)",
    "Tensile_Load_21D(KN)",
    "Tensile_Load_28D(KN)",
    "Tensile_Strength_7D(N/mm2)",
    "Tensile_Strength_14D(N/mm2)",
    "Tensile_Strength_21D(N/mm2)",
    "Tensile_Strength_28D(N/mm2)"
]

tens_df_new_ordr = tens_df.copy()

# Reorder the DataFrame
tens_df_no = tens_df_new_ordr[new_order]

# Display the first few rows
print("First 5 rows of the dataset:")
tens_df_no.head()

# Display all the column names
print("All column names are:")
print(tens_df_no.columns)

# Checking the distribution of Tensile strength
sns.distplot(tens_df_no['Tensile_Load_7D(KN)'],color='green')

# List of days
days = ["7D", "14D", "21D", "28D"]

# Loop over each day and plot both Tensile Load and Strength
for day in days:
    # Plot Tensile Load
    load_col = f"Tensile_Load_{day}(KN)"
    if load_col in tens_df_no.columns:
        sns.histplot(tens_df_no[load_col], kde=True, color='green')
        plt.title(f"Distribution of {day} Tensile Load")
        plt.xlabel("Tensile Load (KN)")
        plt.ylabel("Frequency")
        plt.show()

    # Plot Tensile Strength
    strength_col = f"Tensile_Strength_{day}(N/mm2)"
    if strength_col in tens_df_no.columns:
        sns.histplot(tens_df_no[strength_col], kde=True, color='red')
        plt.title(f"Distribution of {day} Tensile Strength")
        plt.xlabel("Tensile Strength (N/mm2)")
        plt.ylabel("Frequency")
        plt.show()

correlation = tens_df_no['Tensile_Load_7D(KN)'].corr(tens_df_no['Tensile_Strength_7D(N/mm2)'])
print("Correlation between 7D Load and Strength:", correlation)

correlation = tens_df_no['Tensile_Load_14D(KN)'].corr(tens_df_no['Tensile_Strength_14D(N/mm2)'])
print("Correlation between 14D Load and Strength:", correlation)

correlation = tens_df_no['Tensile_Load_21D(KN)'].corr(tens_df_no['Tensile_Strength_21D(N/mm2)'])
print("Correlation between 21D Load and Strength:", correlation)

correlation = tens_df_no['Tensile_Load_28D(KN)'].corr(tens_df_no['Tensile_Strength_28D(N/mm2)'])
print("Correlation between 28D Load and Strength:", correlation)

# Copy
tens_df_new_ordr = tens_df_no.copy()

# Drop the Tensile Load
cols_to_drop = [col for col in tens_df_new_ordr.columns if col.startswith("Tensile_Load_")]
tens_df_new_ordr = tens_df_new_ordr.drop(columns=cols_to_drop)


# Display the first few rows
print("First 5 rows of the dataset:")
tens_df_new_ordr.head()

"""When the load and compressive strength are **perfectly** or **almost perfectly correlated**, we should drop one to avoid **multicollinearity**.
Using **both** would **not improve** the model — it may actually hurt **generalization** or cause **overfitting**.

#### 🎯 Why We Should Drop Redundant Features
##### Problems with keeping both:
- Multicollinearity: Especially affects linear models, but also influences neural networks.
- Increased training time with no benefit.
- Overfitting risk: The model might memorize instead of generalizing.

##### Benefits of dropping:
- Faster and more efficient training.
- Better generalization.
- Easier model interpretability.

#### So, what should we do?
##### ✅ Recommendation:
Drop either **Load** or **Compressive Strength**, depending on:
- What you're trying to predict.
- Which feature is **more directly useful** or **physically meaningful**.
- If you're predicting **Compressive Strength**, then keep it and drop **Load**.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Define the actual Tensile strength feature names
continuous_feature = [
    'Tensile_Strength_7D(N/mm2)',
    'Tensile_Strength_14D(N/mm2)',
    'Tensile_Strength_21D(N/mm2)',
    'Tensile_Strength_28D(N/mm2)'
]

# Plot and save histograms for each feature
for feature in continuous_feature:
    sns.histplot(tens_df_new_ordr[feature], bins=25, kde=True, color='skyblue')
    plt.title(f"Distribution of {feature}")
    plt.xlabel("Strength (N/mm²)")
    plt.ylabel("Count")
    plt.grid(False)
    plt.tight_layout()

    # Clean the feature name to create a valid filename
    clean_feature = feature.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
    filename = f"Histogram_{clean_feature}.png"

    # Save the figure
    plt.savefig(filename)
    plt.show()

# Statistical description of the dataset
print("\nStatistical Description of the Dataset:")
tens_df_new_ordr.describe()

"""### **Correlation**"""

# Copy
tens_df_nw_odr = tens_df_new_ordr.copy()

# Drop or Exclude Non-Numeric Columns
numeric_data = tens_df_nw_odr.select_dtypes(include=['float64', 'int64'])

# Checking the correlation of the dataset
corr = numeric_data.corr()

# Create the heatmap with enhancements
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,
    square=True,
    annot_kws={"size": 8, "weight": "bold"},
)
plt.title("Enhanced Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.xticks(fontsize=10, rotation=45, ha="right", weight="bold")
plt.yticks(fontsize=10, weight="bold")
plt.tight_layout()
plt.savefig("Correlation Heatmap of Compressive Strength")
plt.show()

# Focus on correlation with the target variable
target_col = 'Tensile_Strength_28D(N/mm2)'

# Drop self-correlation and sort by absolute correlation
correlation_with_target = corr[target_col].drop(target_col).abs().sort_values(ascending=False)

# Convert to percentage
percentages = correlation_with_target * 100

# Use a bold colormap: 'coolwarm'
norm = plt.Normalize(percentages.min(), percentages.max())
cmap = plt.cm.coolwarm  # strong from orange-red to deep blue
colors = cmap(norm(percentages.values))

# Moderate figure size
plt.figure(figsize=(7, 5))

# Bar plot
bars = plt.barh(correlation_with_target.index, percentages, color=colors)
plt.gca().invert_yaxis()

# Titles and labels
plt.title(f'Feature Correlation with {target_col}', fontsize=13, fontweight='bold')
plt.xlabel('Absolute Correlation (%)', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Text on bars
for bar, percentage in zip(bars, percentages):
    plt.text(
        bar.get_width() - 2,
        bar.get_y() + bar.get_height() / 2,
        f'{percentage:.1f}%',
        ha='right', va='center',
        color='white', fontsize=9, fontweight='bold'
    )

# Layout and save
plt.tight_layout()
plt.savefig('correlation_feature_importance.png', dpi=150)
plt.show()

# Define the strength columns you want to explore
strength_columns = [
    'Tensile_Strength_7D(N/mm2)',
    'Tensile_Strength_14D(N/mm2)',
    'Tensile_Strength_21D(N/mm2)',
    'Tensile_Strength_28D(N/mm2)'
]

# Plot violin plots for each strength column
for col in strength_columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=tens_df_new_ordr, x='Sample Type', y=col, palette='Set2')
    plt.title(f"Effect of Sample Type on {col}")
    plt.xlabel("Sample Type")
    plt.ylabel("Compressive Strength (N/mm²)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Clean the feature name to create a valid filename
    clean_feature = col.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
    filename = f"Violinplot_{clean_feature}.png"

    # Save the figure
    plt.savefig(filename)
    plt.show()

# Plot box plots for each strength column
for col in strength_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=tens_df_new_ordr, x='Sample Type', y=col, palette='Set3')
    plt.title(f"Effect of Sample Type on {col}")
    plt.xlabel("Sample Type")
    plt.ylabel("Tensile Strength (N/mm²)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Clean the feature name to create a valid filename
    clean_feature = col.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
    filename = f"Boxplot_{clean_feature}.png"

    # Save the figure
    plt.savefig(filename)
    plt.show()

"""## **Data Preprocessing**"""

# Import necessary Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.preprocessing import RobustScaler

df = tens_df_no.copy()

# One-hot encode "Sample Type"
# This will create new columns like "Sample Type_Control", "Sample Type_0.5% H", etc.
df = pd.get_dummies(df, columns=['Sample Type'], drop_first=True)

print("First 5 rows of the dataset:")
tens_df_no.head()

# Define features (X) and target (y)
X = df.drop(['Tensile_Strength_28D(N/mm2)', "Tensile_Load_28D(KN)"], axis=1)
y = df['Tensile_Strength_28D(N/mm2)']

# Handling outliers by replacing them with IQR bounds
# Iterate only over numeric columns for outlier capping
numeric_cols = X.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Using .loc to avoid SettingWithCopyWarning and ensure assignment works correctly
    X.loc[X[col] < lower_bound, col] = lower_bound
    X.loc[X[col] > upper_bound, col] = upper_bound

# Standardize features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Features (X):")
print(X_scaled.head())
print("\nTarget (y):")
print(y.head())

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\nDataset split completed:")
print(f"Total samples: {X_scaled.shape[0]}, Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# checking the number of Test and Train dataset
print(X_scaled.shape, X_train.shape, X_test.shape)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)
print("RF (capped+scaled) R²:", rf.score(X_test, y_test))

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression # Import RidgeCV and LinearRegression
from xgboost import XGBRegressor # Assuming you have xgboost installed

estimators = [
    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
]

stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stack.fit(X_train, y_train)
print("Stacked R²:", stack.score(X_test, y_test))

"""## **Build ANN with Hyperparameter Tuning**"""

# Importing tensorflow and Keras
import tensorflow as tf  #  Imports TensorFlow for deep learning tasks.
tf.random.set_seed(42)    # s Ensures reproducibility by setting a fixed random seed.
from tensorflow import keras # Imports Keras, a high-level API for building models.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Build a small-to-moderate ANN with Dropout + BatchNorm
def build_model(input_dim,
                hidden_layers: int = 2,
                neurons: int = 32,
                learning_rate: float = 1e-3,
                dropout_rate: float = 0.2):
    """
    - input_dim: number of features (= 11 in your case)
    - hidden_layers: how many Dense layers (excluding the output)
    - neurons: units per hidden layer
    - learning_rate: for Adam
    - dropout_rate: fraction to drop after each Dense+BatchNorm block
    """
    model = Sequential()
    # Explicit Input layer
    model.add(Input(shape=(input_dim,)))

    # First hidden layer (Dense → BatchNorm → Dropout)
    model.add(Dense(neurons, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Possibly additional hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer (1 unit, linear)
    model.add(Dense(1, activation='linear'))

    # Compile with Adam + MSE loss + track MAE
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model

# Hyperparameters to experiment with:
chosen_hidden_layers = 2
chosen_neurons       = 32
chosen_lr            = 1e-3
chosen_dropout_rate  = 0.2
chosen_epochs        = 200
chosen_batch_size    = 16

model = build_model(
    input_dim=X_train.shape[1],
    hidden_layers=chosen_hidden_layers,
    neurons=chosen_neurons,
    learning_rate=chosen_lr,
    dropout_rate=chosen_dropout_rate
)

model.summary()

# Set up EarlyStopping & Train
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,             # if val_loss doesn’t improve for 15 epochs, stop
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,   # hold out 15% of train for validation
    epochs=chosen_epochs,
    batch_size=chosen_batch_size,
    callbacks=[early_stop],
    verbose=1
)

# Plot Training / Validation Curves
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Train vs. Val Loss')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Train vs. Val MAE')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate on Test Set
y_pred = model.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred)
test_r2  = r2_score(y_test, y_pred)

print(f"\nFinal Test MSE: {test_mse:.3f}")
print(f"Final Test R² : {test_r2:.3f}")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Linear Regression R²:", lr.score(X_test, y_test))
print("Random Forest R²:", rf.score(X_test, y_test))

"""## **Evaluate on Test Set**"""

