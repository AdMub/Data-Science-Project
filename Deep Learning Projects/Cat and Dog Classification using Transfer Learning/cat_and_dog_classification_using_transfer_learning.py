# -*- coding: utf-8 -*-
"""Cat and Dog Classification using Transfer Learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19RWp9hjgdqJeYugT7MhJpUm2oG7uFlfl

#**Dogs vs. Cats**

##**Overview**


## **Description**
In this competition, you'll write an algorithm to classify whether images contain either a dog or a cat.  This is easy for humans, dogs, and cats. Your computer will find it a bit more difficult.

![Description](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3362/media/woof_meow.jpg)



           Deep Blue beat Kasparov at chess in 1997.

    Watson beat the brightest trivia minds at Jeopardy in 2011.

        Can you tell Fido from Mittens in 2013?



##**The Asirra data set**
Web services are often protected with a challenge that's supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute-force attacks on web site passwords.

Asirra (Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. Many even think it's fun! Here is an example of the Asirra interface:

Asirra is unique because of its partnership with Petfinder.com, the world's largest site devoted to finding homes for homeless pets. They've provided Microsoft Research with over three million images of cats and dogs, manually classified by people at thousands of animal shelters across the United States. Kaggle is fortunate to offer a subset of this data for fun and research.

##**Image recognition attacks**
While random guessing is the easiest form of attack, various forms of image recognition can allow an attacker to make guesses that are better than random. There is enormous diversity in the photo database (a wide variety of backgrounds, angles, poses, lighting, etc.), making accurate automatic classification difficult. In an informal poll conducted many years ago, computer vision experts posited that a classifier with better than 60% accuracy would be difficult without a major advance in the state of the art. For reference, a 60% classifier improves the guessing probability of a 12-image HIP from 1/4096 to 1/459.

##**State of the art**
The current literature suggests machine classifiers can score above 80% accuracy on this task [1]. Therfore, Asirra is no longer considered safe from attack.  We have created this contest to benchmark the latest computer vision and deep learning approaches to this problem. Can you crack the CAPTCHA? Can you improve the state of the art? Can you create lasting peace between cats and dogs?

Okay, we'll settle for the former.



##**Acknowledgements**
We extend our thanks to Microsoft Research for providing the data for this competition.

* Jeremy Elson, John R. Douceur, Jon Howell, Jared Saul, Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization, in Proceedings of 14th ACM Conference on Computer and Communications Security (CCS), Association for Computing Machinery, Inc., Oct. 2007

#**Extracting Dataset using Kaggle API**
"""

# Install the Kaggle Library
!pip install Kaggle

# Configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

"""**Install necessary Libraries**"""

!pip install tensorflow==2.15.0 tensorflow-hub keras

pip install gradio

"""#**Import the Libraries**"""

import numpy as np
import pandas as pd
import os
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow

"""#**Importing the Datasets from Kaggle**"""

# Kaggle API
!kaggle competitions download -c dogs-vs-cats

!ls

# Extracting the compressed dataset
from zipfile import ZipFile
file_name = "/content/dogs-vs-cats.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')

# Extracting the compressed dataset
from zipfile import ZipFile
dataset = "/content/train.zip"
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('Done')

# Counting the number of files in train folder
path, dirs, files = next(os.walk("/content/train"))
file_count = len(files)
print("Number of Images:", file_count)

"""**Printing the name of Images**"""

# Printing the name of images
for i in os.listdir("/content/train"):
  print(i)

# Printing the name of images
file_names = os.listdir("/content/train")
print(file_names)

"""#**Display the Images of Cats and Dogs**"""

# Display the cat image
img = mpimg.imread('/content/train/cat.7517.jpg')
imgplot = plt.imshow(img)
plt.show()

# Display the dog image
img = mpimg.imread('/content/train/dog.5233.jpg')
imgplot = plt.imshow(img)
plt.show()

# Checking the number of Cats Images
for i in os.listdir("/content/train"):
  if i.startswith("cat"):
    print(i)

file_names = os.listdir("/content/train")

for i in range(5):
  name = file_names[i]
  print(name)

file_names = os.listdir("/content/train")

for i in range(5):
  name = file_names[i]
  print(name[0:3])

file_names = os.listdir("/content/train")

cat_count = 0
dog_count = 0

for img_file in file_names:
  img_name = img_file[0:3]
  if img_name == "cat":
    cat_count += 1
  else:
    dog_count += 1

print("Number of Cat Images:", cat_count)
print("Number of Dog Images:", dog_count)

# Initialize a counter for cat images
cat_count = 0

# Iterate through files in the directory
for i in os.listdir("/content/train"):
  # Check if the filename starts with "cat"
  if i.startswith("cat"):
    # Increment the counter if it's a cat image
    cat_count += 1

# Print the total count of cat images
print("Number of Cat Images:", cat_count)

# Initialize a counter for cat images
dog_count = 0

# Iterate through files in the directory
for i in os.listdir("/content/train"):
  # Check if the filename starts with "cat"
  if i.startswith("dog"):
    # Increment the counter if it's a cat image
    dog_count += 1

# Print the total count of cat images
print("Number of Dog Images:", dog_count)

"""**Resizing all the Images**"""

# Creating a directory for resized Images
os.mkdir("/content/resized_images")

original_folder = "/content/train/"
resized_folder = "/content/resized_images/"

for i in range(3000):

  filename = os.listdir(original_folder)[i]
  image_path = original_folder+filename

  img = Image.open(image_path)
  img = img.resize((224, 224))
  img.convert("RGB")

  new_image_path = resized_folder+filename
  img.save(new_image_path)

# Display the resized cat image
img = mpimg.imread('/content/resized_images/cat.8463.jpg')
imgplot = plt.imshow(img)
plt.show()

# Display the resized dog image
img = mpimg.imread('/content/resized_images/dog.5675.jpg')
imgplot = plt.imshow(img)
plt.show()

"""**Creating labels for resized images for dogs and cats**

*  **Cat ---> 0**
*  **Dog ---> 1**
"""

# Creating a for loop to assign labels to resized images
file_names = os.listdir("/content/resized_images")

labels = []

for i in range(3000):

  img_name = file_names[i]
  if img_name[0:3] == "cat":
    labels.append(0)
  else:
    labels.append(1)

print(file_names[0:10])
print(len(file_names))

print(labels[0:10])
print(len(labels))

# Counting the images of dogs and cats out of 2000 images
values, counts = np.unique(labels, return_counts=True)
print(values)
print (counts)

# Counting the images of dogs and cats out of 5000 images
cat_count = 0
dog_count = 0

for i in labels:
  if i == 0:
    cat_count += 1
  else:
    dog_count += 1

print("Number of Cat Images:", cat_count)
print("Number of Dog Images:", dog_count)

# Plotting a plot distribution for cat and dogs from the 5000 images
sns.set()
# Create a Pandas DataFrame from the labels list
df = pd.DataFrame({'label': labels})
# Map the numerical labels to 'cat' and 'dog'
df['label'] = df['label'].map({0: 'cat', 1: 'dog'})
# Now use the DataFrame in countplot
sns.countplot(x='label', data=df)
plt.show() # Add this line to display the plot

plot = plt.figure(figsize=(5,5))
# Use the 'label' column for x-axis and count for the y-axis
# The estimator argument is set to 'count' (default) to calculate the frequency of each label.
sns.countplot(data=df, x='label')
plt.show()

"""**Converting all the resized images to numpy arrays**"""

import cv2
import glob

# Specify the directory where images are stored
image_directory = "/content/resized_images/"

# Define supported image formats
image_extension = ["png", "jpg"]

# Initialize an empty list to store image file paths
files = []

# Search for image files in the directory and add them to the list
[files.extend(glob.glob(image_directory + "*." + e)) for e in image_extension]

# Check if images were found
if not files:
    print("No images found in the directory.")
else:
    # Read images using OpenCV and convert to a NumPy array
    dog_cat_images = np.array([cv2.imread(file) for file in files])

    # Print the shape of the resulting array (if images are the same size)
    print("Loaded images shape:", dog_cat_images.shape)

print(dog_cat_images)

type(dog_cat_images)

X = dog_cat_images
y = np.asarray(labels)

"""# **Splitting the Dataset into Training and Test Sets**"""

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
print("\nDataset split completed:")
print(f"Total samples: {X.shape[0]}, Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

print(X.shape)
print(X_train.shape)
print(X_test.shape)

"""* **3750 --> Training Images**
* **1250 --> Test Images**
"""

# Normalization (Scaling the values to be in range of 0 - 1)
X_train_norm = X_train / 255.0

X_test_norm = X_test / 255.0

print(X_train_norm)

"""# **Building the Neural Network**"""

import tensorflow as tf
import tensorflow_hub as hub

mobilenet_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224, 224, 3), trainable=False)

num_of_classes = 2

model = tf.keras.Sequential([

    pretrained_model,

    tf.keras.layers.Dense(num_of_classes)
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(X_train_norm, y_train, validation_split=0.2, epochs=10)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Data', 'Validation Data'], loc='lower right')
plt.show

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Data', 'Validation Data'], loc='upper right')
plt.show

loss, accuracy = model.evaluate(X_test_norm, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

model = model.save("trained_model.h5")

"""# **Predictive System**"""

# Enter the path of Image
input_image_path = input("Path of the image to be predicted: ")

# Read the image entered
input_image = cv2.imread(input_image_path)

# Display the Image
cv2_imshow(input_image)

# Resize the image to 224x224 pixels
input_image_resize = cv2.resize(input_image, (224, 224))

# Normalization (Scaling the values to be in range of 0 - 1)
input_image_scaled = input_image_resize / 255

# Reshaping the image
image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

# Predicting the image
input_prediction = model.predict(image_reshaped)

# Get the class label with the highest probability
input_pred_label = np.argmax(input_prediction)

print(input_prediction)
print(input_pred_label)

if input_pred_label == 0:
  print("The image representation is a cat")
else:
  print("The image representation is a dog")

# Enter the path of Image
input_image_path = input("Path of the image to be predicted: ")

# Read the image entered
input_image = cv2.imread(input_image_path)

# Display the Image
cv2_imshow(input_image)

# Resize the image to 224x224 pixels
input_image_resize = cv2.resize(input_image, (224, 224))

# Normalization (Scaling the values to be in range of 0 - 1)
input_image_scaled = input_image_resize / 255

# Reshaping the image
image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

# Predicting the image
input_prediction = model.predict(image_reshaped)

# Get the class label with the highest probability
input_pred_label = np.argmax(input_prediction)

print(input_prediction)
print(input_pred_label)

if input_pred_label == 0:
  print("The image representation is a cat")
else:
  print("The image representation is a dog")

# Enter the path of Image
input_image_path = input("Path of the image to be predicted: ")

# Read the image entered
input_image = cv2.imread(input_image_path)

# Display the Image
cv2_imshow(input_image)

# Resize the image to 224x224 pixels
input_image_resize = cv2.resize(input_image, (224, 224))

# Normalization (Scaling the values to be in range of 0 - 1)
input_image_scaled = input_image_resize / 255

# Reshaping the image
image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

# Predicting the image
input_prediction = model.predict(image_reshaped)

# Get the class label with the highest probability
input_pred_label = np.argmax(input_prediction)

print(input_prediction)
print(input_pred_label)

if input_pred_label == 0:
  print("The image representation is a cat")
else:
  print("The image representation is a dog")

# Enter the path of Image
input_image_path = input("Path of the image to be predicted: ")

# Read the image entered
input_image = cv2.imread(input_image_path)

# Display the Image
cv2_imshow(input_image)

# Resize the image to 224x224 pixels
input_image_resize = cv2.resize(input_image, (224, 224))

# Normalization (Scaling the values to be in range of 0 - 1)
input_image_scaled = input_image_resize / 255

# Reshaping the image
image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

# Predicting the image
input_prediction = model.predict(image_reshaped)

# Get the class label with the highest probability
input_pred_label = np.argmax(input_prediction)

print(input_prediction)
print(input_pred_label)

if input_pred_label == 0:
  print("The image representation is a cat")
else:
  print("The image representation is a dog")

import gradio as gr

from tensorflow.keras.models import load_model

# Define a dictionary mapping the custom layer name to its implementation
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the trained model using 'custom_objects'
model = load_model("trained_model.h5", custom_objects=custom_objects)  # Ensure your model is saved and available

def predict_image(input_image):
    """Predict if the uploaded image is a cat or a dog."""

    # Convert image to OpenCV format (BGR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Resize the image to 224x224 pixels
    input_image_resize = cv2.resize(input_image, (224, 224))

    # Normalize pixel values (scale between 0 - 1)
    input_image_scaled = input_image_resize / 255.0

    # Reshape the image to match the model's expected input shape
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

    # Predict the class
    input_prediction = model.predict(image_reshaped)

    # Get the class label with the highest probability
    input_pred_label = np.argmax(input_prediction)

    # Define class labels
    class_labels = ["Cat", "Dog"]

    return f"The image representation is a {class_labels[input_pred_label]}"

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),  # Accepts image input
    outputs="text",  # Outputs text classification
    title="🐱🐶 Cat vs Dog Classifier",
    description="Upload an image of a cat or a dog, and the model will classify it."
)

# Launch the Gradio App
interface.launch()

def predict_image(input_image):
    """Predict if the uploaded image is a cat or a dog."""

    # Convert image to OpenCV format (BGR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Resize the image to 224x224 pixels
    input_image_resize = cv2.resize(input_image, (224, 224))

    # Normalize pixel values (scale between 0 - 1)
    input_image_scaled = input_image_resize / 255.0

    # Reshape the image to match the model's expected input shape
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

    # Predict the class
    input_prediction = model.predict(image_reshaped)

    # Get the class label with the highest probability
    input_pred_label = np.argmax(input_prediction)

    # Define class labels
    class_labels = ["Cat", "Dog"]

    return f"The image representation is a {class_labels[input_pred_label]}"

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),  # Accepts image input
    outputs="text",  # Outputs text classification
    title="🐱🐶 Cat vs Dog Classifier",
    description="Upload an image of a cat or a dog, and the model will classify it."
)

# Launch the Gradio App
interface.launch()