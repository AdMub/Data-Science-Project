# 🌿 PlantVillage: CNN-Based Plant Disease Detection

![image](https://github.com/user-attachments/assets/e79e0571-207d-423f-a54f-656c11c37489)


A deep learning project designed to detect and classify **plant leaf diseases** using a custom **Convolutional Neural Network (CNN)** trained on the **PlantVillage Dataset**.

---

## 📌 Project Overview

Agricultural crop diseases significantly impact food security and farmer livelihoods. This project aims to automate the identification of plant diseases using computer vision, enabling fast and accurate diagnosis through images of plant leaves.

Using the publicly available **PlantVillage dataset**, the model classifies leaves into **38 classes**, including both healthy and diseased categories.

---

## 🎯 Objectives

- Build a Convolutional Neural Network to classify plant diseases from leaf images.
- Evaluate model performance on unseen data.
- Deploy the model with an interactive web interface for real-time predictions (Gradio).

---

## 🗃️ Dataset

- **Source**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Images**: 50,000+ leaf images
- **Categories**: 38 (various diseases + healthy conditions)
- **Formats**: Color, grayscale, and segmented versions

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Image resizing and normalization
   - Training-validation split (80-20)

2. **Model Architecture**
   - Custom CNN with Conv2D, MaxPooling, and Dense layers
   - Output layer with Softmax activation for multiclass classification

3. **Training**
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy, Precision, Recall

4. **Evaluation**
   - Plotted learning curves
   - Assessed overfitting using validation metrics

5. **Deployment**
   - Developed a simple Gradio interface for real-time predictions
   <img width="722" alt="plant_disea_output" src="https://github.com/user-attachments/assets/964662f6-602b-402f-ae5d-a436fc3e7d16" />
   <img width="722" alt="plant_disea_output" src="https://github.com/user-attachments/assets/5cfbfd19-552a-4e20-8e91-c9ac82a8acc4" />




---

## 🧪 Results

- **Training Accuracy**: ~98%  
- **Validation Accuracy**: ~96%  
- **Model Generalization**: Good performance with minimal overfitting
- **Deployment**: Successfully deployed with Gradio for demo/testing

---

## 🛠️ Tools & Technologies

- Python (TensorFlow, Keras, NumPy, Matplotlib)
- Jupyter Notebook / Google Colab
- Gradio (for interactive deployment)
- Kaggle API (for dataset download)

---

## 🚀 How to Use

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or Python script
4. Launch the Gradio interface to make predictions with your own leaf images

---

## 📚 References

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419.
- [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [Gradio Documentation](https://www.gradio.app/)

---

## 👤 Author
- Mubarak Adisa
- AI & Deep Learning Enthusiast | Civil & Environmental Engineer
- 🔗 GitHub: [AdMub](https://github.com/AdMub)

---

## ⭐️ Show your support
If you found this helpful, please ⭐️ the repo and share your thoughts!
