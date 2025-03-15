# 🏥 Face Mask Detection using Deep Learning 😷

![Face Mask Detection](image_link_here)  <!-- Replace with an actual image link -->

## 📌 Project Overview
This project implements a **Face Mask Detection System** using **Deep Learning** and **Convolutional Neural Networks (CNNs)**. The model is trained on **7,553 labeled images** to classify whether a person is **wearing a mask or not**.

The trained model achieves:
✅ **94% Training Accuracy**  
✅ **96% Validation Accuracy**

## 📂 Dataset Details
The dataset consists of **7,553 RGB images** in two categories:
- 🟢 **With Mask** → **3,725 images**
- 🔴 **Without Mask** → **3,828 images**

### **📌 Dataset Sources**
- **1,776 images** were obtained from **Prajna Bhandary's GitHub Repository**:  
  🔗 [GitHub Repository](https://github.com/prajnasb/observations)  
- **5,777 additional images** were collected from **Google Search**.

---

## 📥 Extracting Dataset using Kaggle API
To download the dataset, configure the **Kaggle API**:
```sh
pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then, download and extract the dataset:
```sh
!kaggle datasets download omkargurav/face-mask-dataset
unzip face-mask-dataset.zip
```

## 🔧 Install Dependencies
Before running the project, install the required Python libraries:
```sh
pip install tensorflow keras numpy pandas matplotlib opencv-python gradio
```

## 📊 Data Preprocessing
### ✅ Image Processing Steps
✔ **Resizing** → All images resized to **128×128 pixels**
✔ **Normalization** → Pixel values scaled to **0-1**
✔ **Conversion to NumPy arrays**
✔ **Splitting Dataset → 75% Training, 25% Testing**

```python
# Convert images to numpy array
data, labels = [], []
for category, label in zip(["with_mask", "without_mask"], [1, 0]):
    for img_file in os.listdir(f"/content/data/{category}"):
        image = Image.open(f"/content/data/{category}/{img_file}")
        image = image.resize((128, 128)).convert("RGB")
        data.append(np.array(image))
        labels.append(label)

X = np.array(data) / 255.0  # Normalize pixel values
y = np.array(labels)
```

## 🤖 Building & Training the CNN Model
### ✅ CNN Architecture
✔ **Convolutional Layers** → Extract important image features
✔ **MaxPooling** → Reduce spatial dimensions
✔ **Flatten** → Convert feature maps to a dense vector
✔ **Dense Layers** → Fully connected layers for classification


```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)
```

## 📈 Model Evaluation
After training, the model achieves **92.43% accuracy** on test data.

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## 💾 Save & Load the Model
To reuse the trained model:
```python
model.save("face_mask_model.h5")
```

To load it for future use:
```python
from tensorflow.keras.models import load_model
model = load_model("face_mask_model.h5")
```

## 🎯 Real-Time Mask Detection System
### 📌 Command Line Prediction

```python
# Load and preprocess the input image
input_img = cv2.imread("test_image.jpg")
input_img_resized = cv2.resize(input_img, (128,128)) / 255.0
input_img_reshaped = np.reshape(input_img_resized, [1,128,128,3])

# Make a prediction
input_pred = model.predict(input_img_reshaped)
input_pred_label = np.argmax(input_pred)

# Print result
if input_pred_label == 1:
    print("✅ Wearing a mask")
else:
    print("❌ Not wearing a mask")
```

## 🌐 Deploying with Gradio
This **Gradio** web app allows users to **upload an image** and receive a prediction.

```python
import gradio as gr

def predict_mask(image):
    input_img_resized = cv2.resize(image, (128, 128)) / 255.0
    input_img_reshaped = np.reshape(input_img_resized, [1, 128, 128, 3])
    input_pred = model.predict(input_img_reshaped)
    input_pred_label = np.argmax(input_pred)

    return "✅ Wearing a Mask" if input_pred_label == 1 else "❌ Not Wearing a Mask"

# Gradio Interface
interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Face Mask Detection 😷",
    description="Upload an image to check if the person is wearing a mask or not."
)

interface.launch()
```

## 🎯 Key Features
✅ **Deep Learning-Based Mask Detection**
✅ **Real-Time Image Processing**
✅ **High Accuracy Model (92.43%)**
✅ **Web Interface Using Gradio**
✅ **Trained on 7,553 Images**

## 📌 Future Improvements
🔹 Train with **larger dataset** for better generalization
🔹 Use **Transfer Learning** (ResNet, MobileNet) for higher accuracy
🔹 Deploy on **Flask** or **FastAPI** for real-world use
🔹 Implement **real-time video mask** detection using OpenCV


## ✨ Credits & Acknowledgements
* **Dataset:** Kaggle & Google Images
* **CNN Implementation:** TensorFlow/Keras
* **Web App:** Gradio

## 📩 Connect & Collaborate
🔗 **GitHub**: [AdMub](https://github.com/AdMub)  
📧 **Email**: [admub465@gmail.com](mailto:admub465@gmail.com)  
🚀 **LinkedIn**: [Mubarak Adisa](https://www.linkedin.com/in/mubarak-adisa-334a441b6/)  

---

### **📝 Key Features of This README**
✔ **Well-Structured** → Clear Sections (Overview, Dataset, Training, Evaluation, Deployment, Future Work)  
✔ **Code Snippets** → Ready to Copy-Paste for Quick Setup  
✔ **Web App Implementation** → Deploy Using **Gradio**  
✔ **Professional Formatting** → Markdown Syntax for a Clean Look  
✔ **Social Links** → Connect for Collaboration  

---
Let me know if you need any modifications! 🚀😊
