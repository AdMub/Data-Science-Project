# 🧠 MNIST Handwritten Digit Generation using DCGAN

<div align="center">
  <img src="https://github.com/user-attachments/assets/594e6575-c233-41e3-bf2b-00cd442cbc66" alt="Generated Digits" width="600"/>
</div>

---

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic images of handwritten digits based on the **MNIST dataset**.

<div align="center">
  <img src="https://raw.githubusercontent.com/AdMub/Data-Science-Project/main/Deep_Learning_Projects/MNIST_DCGAN/images/dcgan.gif" alt="Generated Digits" width="600"/>
</div>

---

## 🚀 Project Overview

The goal is to explore deep generative modeling through GANs, particularly the DCGAN architecture, by training a generator and discriminator in a competitive setting to create convincing digit images.

---

## 🧱 DCGAN Architecture Overview

Below is a visual summary of how the Generator and Discriminator interact during training.

<img src="https://raw.githubusercontent.com/AdMub/Data-Science-Project/main/Deep_Learning_Projects/MNIST_DCGAN/images/Generative Adversarial Network Architecture.jpg" alt="DCGAN Architecture" width="600"/>

---

## 🧰 Technologies Used

- 🧠 **PyTorch** — deep learning framework for model development
- 🧮 **NumPy & Matplotlib** — for data manipulation and visualization
- 🖼️ **MNIST Dataset** — standard dataset of 28x28 grayscale digit images
- 🎨 **DCGAN Architecture** — convolutional neural networks in GANs
- 📈 **GIF Generation** — visualize training progress of digit generation

---

## 📁 Project Structure

```plain
MNIST_DCGAN/
├── dcgan.gif # Training progress animation
├── images/ # Generated samples at intervals
├── README.md # Project documentation
├── dcgan_mnist.ipynb # Jupyter notebook (optional)
├── dcgan_mnist.py # Python script (optional)
└── saved_models/ # Checkpoints (if saved)
```


---

## 🧠 How DCGAN Works

- **Generator**: learns to create images similar to MNIST digits from random noise.
- **Discriminator**: distinguishes real MNIST images from fake ones.
- Training is adversarial — the generator tries to fool the discriminator.

---

## 🧪 Results

The generator improves over time and produces increasingly realistic handwritten digits.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2e0e20cb-ed9a-46a0-b770-a95f5dd7c779" alt="Generated Digits" width="400"/>
</div>

---

## ⚙️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AdMub/Data-Science-Project.git
   cd Data-Science-Project/Deep_Learning_Projects/MNIST_DCGAN
   ```


2. **Install dependencies**  
   ```bash
    pip install torch torchvision matplotlib
    ```

3. **Run the project**

- Jupyter Notebook: Open --dcgan_mnist.ipynb--
- Python script:

    ```bash
        python dcgan_mnist.py
     ```

## **📚 References**
- [TensorFlow DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661) — Original GAN paper
- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## **🙋‍♂️ Author**
Mubarak Adisa(AdMub)

📧 admub465@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/mubarak-adisa-334a441b6/)

📂 [GitHub Portfolio](https://github.com/AdMub)

## **🌟 Acknowledgements**
Special thanks to the Sidhardhan, deep learning community and open-source contributors for enabling this educational project.
