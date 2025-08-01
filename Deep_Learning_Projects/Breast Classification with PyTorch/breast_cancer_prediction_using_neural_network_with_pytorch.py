# -*- coding: utf-8 -*-
"""Breast Cancer Prediction using Neural Network with Pytorch.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jWAci-_MUbsQOsUa0oNKD75eT5qx2PT5

###**Importing necessary Libraries**
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""###**Device Configuration**"""

# check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

print(X)

print(y[:20])

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Standardizing the data using Standard scaler to avoid data leakage and normally distributed
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

type(X_train)

# Convert data to PyTorch tensors and move it to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

"""###**Building Neural Network Architecture**"""

# Define the neural network architecture
class NeuralNet(nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.fc1  = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.sigmoid(out)
    return out

print(X_train.shape[1])

# Define the hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

# Initiliaze the neural network and move it to the GPU
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define the loss and the optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""###**Training the Neural Network**"""

# Training the model
for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = criterion(outputs, y_train.view(-1, 1))
  loss.backward()
  optimizer.step()

  # Evaluate accuracy
  with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    predicted = (outputs.round() > 0.5).float()
    acc = (predicted == y_test.view(-1, 1)).float().mean()
    correct = (predicted == y_test.view(-1, 1)).float().sum()
    accuracy = correct / y_test.size(0)
    total = y_test.size(0)

  if (epoch+1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%")

"""###**Model Evaluation**"""

# Evaluation on Training set
model.eval()
with torch.no_grad():
  outputs = model(X_train)
  predicted = outputs.round()
  accuracy = (predicted == y_train.view(-1, 1)).float().sum()
  accuracy = accuracy / y_train.size(0)
  print(f"Training Accuracy: {accuracy.item()*100:.2f}%")

# Evaluation on Training set
model.eval()
with torch.no_grad():
  outputs = model(X_train)
  predicted = outputs.round()
  accuracy = (predicted == y_train.view(-1, 1)).float().mean()
  print(f"Training Accuracy: {accuracy.item()*100:.2f}%")

# Evaluation on Test set
model.eval()
with torch.no_grad():
  outputs = model(X_test)
  predicted = outputs.round()
  accuracy = (predicted == y_test.view(-1, 1)).float().sum()
  accuracy = accuracy / y_test.size(0)
  print(f"Test Accuracy: {accuracy.item()*100:.2f}%")

# Evaluation on Test set
model.eval()
with torch.no_grad():
  outputs = model(X_test)
  predicted = outputs.round()
  accuracy = (predicted == y_test.view(-1, 1)).float().mean()
  print(f"Test Accuracy: {accuracy.item()*100:.2f}%")