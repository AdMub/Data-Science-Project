# 🐶🐱 Dogs vs. Cats Classification Using Deep Learning
![Image DEscription](https://github.com/user-attachments/assets/9e9c340d-ad05-4f07-9447-99eb9a095a43)

## 📌 **Project Overview**
This project is a **Deep Learning-based Image Classification** model that can **classify images** as either **a dog 🐶 or a cat 🐱**. The model is trained using the **Dogs vs. Cats dataset** from Kaggle, leveraging a **pretrained MobileNetV2** model for feature extraction.

## 📂 **Dataset Overview**
The dataset consists of **25,000 images** of cats and dogs, equally split (12,500 images per class). It was originally part of the **Asirra CAPTCHA challenge**, where users had to distinguish between dogs and cats.

### **Key Information**
- **Label 0** → **Cat 🐱**
- **Label 1** → **Dog 🐶**
- **Source**: [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)

### **Extracting the Dataset using Kaggle API**
```sh
# Install Kaggle API
pip install kaggle

# Configure Kaggle API (ensure you have kaggle.json file)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and extract dataset
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip
unzip train.zip
```

## 🚀 Installation & Setup
### ** Install Required Libraries**
To run this project, install the required dependencies:
```sh
pip install tensorflow==2.15.0 tensorflow-hub keras numpy pandas matplotlib seaborn gradio opencv-python
```


## 🔧 Data Preprocessing
* **Resize all images to 224x224 pixels** to match MobileNetV2 input.
```sh
import cv2
import os
from PIL import Image

original_folder = "/content/train/"
resized_folder = "/content/resized_images/"
os.makedirs(resized_folder, exist_ok=True)

for filename in os.listdir(original_folder):
    img = Image.open(original_folder + filename)
    img = img.resize((224, 224))
    img.save(resized_folder + filename)
```

* **Normalize pixel values** to range **0-1** and **Convert images into NumPy arrays**.
```sh
import numpy as np
image_directory = "/content/resized_images/"
image_extension = ["png", "jpg"]

files = []
[files.extend(glob.glob(image_directory + "*." + e)) for e in image_extension]

dog_cat_images = np.array([cv2.imread(file) for file in files]) / 255.0
```

## 🏗 Model Architecture
I utilize **MobileNetV2** (a lightweight, high-performance CNN) for feature extraction.
```sh
import tensorflow as tf
import tensorflow_hub as hub

mobilenet_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Dense(2)  # Output layer with 2 classes: Cat & Dog
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
```

## 📉 Model Training
### Train the Model
```sh
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10)
```

### Evaluate the Model
```sh
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```
### Model Performance

Test Accuracy: **97.2%** 🎯


## 🤖 Predictive System
```sh
def predict_image(input_image):
    """Predict if the uploaded image is a cat or a dog."""
    
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    input_image_resize = cv2.resize(input_image, (224, 224))
    input_image_scaled = input_image_resize / 255.0
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    class_labels = ["Cat", "Dog"]
    return f"The image representation is a {class_labels[input_pred_label]}"
```

## 🎨 Gradio Web App
We deploy the model using Gradio for an interactive user interface.
```sh
import gradio as gr
from tensorflow.keras.models import load_model

# Load trained model (handling custom TensorFlow Hub layers)
custom_objects = {"KerasLayer": hub.KerasLayer}
model = load_model("trained_model.h5", custom_objects=custom_objects)

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🐱🐶 Cat vs Dog Classifier",
    description="Upload an image of a cat or a dog, and the model will classify it."
)

# Launch the Web App
interface.launch()
```

## 📌 Conclusion
* Successfully built a **Deep Learning Image Classifier** for cats and dogs.
* Achieved **97.2% accuracy** using **MobileNetV2**.
* Created an **interactive Gradio web app** for easy predictions.

## 📸 Project Image
<img width="612" alt="output2" src="https://github.com/user-attachments/assets/55cf4127-79c2-4fd2-9309-9ed617e19f5c" />

<img width="618" alt="output1" src="https://github.com/user-attachments/assets/b273e675-8e0d-4f58-a885-a2b3230ead02" />


## 💡 Future Improvements
* Implement **Data Augmentation** to improve model generalization.
* Deploy using **Flask/FastAPI** for production.
* Tune **hyperparameters** for better performance.
  
### **📩 Feel free to contribute or reach out for collaborations!** 🚀
