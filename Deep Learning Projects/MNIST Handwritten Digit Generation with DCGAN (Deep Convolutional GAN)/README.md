# 🧠 MNIST Handwritten Digit Generation with DCGAN

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate handwritten digits that resemble those from the MNIST dataset. The generator learns to produce realistic images, while the discriminator learns to distinguish real from fake ones — both improving in a zero-sum game.

![DCGAN Output GIF](dcgan.gif)

---

## 📦 Installation

To run this notebook, you need the following libraries:

```bash
pip install tensorflow imageio tensorflow-docs
```

## 📂 Dataset

We use the MNIST dataset containing 60,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels.

```python
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
```
The images are reshaped and normalized to the range [-1, 1] to fit the tanh activation used in the generator.



## 🧱 Model Architecture
### 🔷 Generator
- Fully connected Dense layer → reshape into 7x7x256
- 3 layers of Conv2DTranspose with BatchNorm and LeakyReLU
- Final output: 28x28x1 grayscale image


```python
def make_generator_model():
    ...
```


### 🔶 Discriminator
- 2 Conv2D layers with LeakyReLU and Dropout
- Flatten + Dense layer to classify real vs. fake

```python
def make_discriminator_model():
    ...
```




## 🧮 Loss Functions
- Generator Loss: Binary cross-entropy comparing fake outputs to 'real' labels.
- Discriminator Loss: Sum of real vs. real labels and fake vs. fake labels.

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```



## 🧰 Optimizers & Checkpoints
Adam optimizers with learning rate 1e-4 are used for both generator and discriminator.

Checkpointing is done every 15 epochs:

```python
checkpoint_dir = './training_checkpoints'
```



## 🔁 Training Loop
- Trains over **100 epochs**
- Saves image samples at each epoch
- Stores checkpoints and visualizes generator progress

```python
def train(dataset, epochs):
    ...
```



## 🖼️ Generated Output & GIF
After training, images are saved and compiled into a GIF:

```python
with imageio.get_writer("dcgan.gif", mode="I") as writer:
    ...
```

The GIF helps you visually track how the model improves across epochs.



## 📊 Results
Here’s an example of generated digits after training:



## ▶️ Running the Project
This notebook is designed for Google Colab or Jupyter Notebook.

- Run all cells from start to finish.
- At the end, a GIF showing generated digit evolution is displayed.
- Checkpoints are saved locally.

## 🎓 What You’ll Learn
- Fundamentals of GANs and DCGAN architecture
- Building custom generator/discriminator models
- Visualizing GAN outputs dynamically
- Saving model checkpoints and generating media





