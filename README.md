# Vanilla-GAN-Image-Generation

This project implements a Vanilla Generative Adversarial Network (GAN) to generate synthetic images that look realistic but do not expose real individuals, enabling privacy-preserving data sharing and augmentation.

****🔹 Module-Wise Implementation**

**Module 1 — Data Pipeline & Preprocessing**
This module prepares raw images for GAN training.
Implemented Features:
Load images in JPG / PNG format (MNIST-style images)
Resize images (e.g., 28×28 or 64×64)
Convert to grayscale or RGB
Normalize pixel values from [0, 255] → [-1, 1] (Tanh compatible)
Batch processing for faster training
Purpose:
Ensures clean, standardized input data for stable GAN training.

**Module 2 — Model Design (Vanilla GAN Architecture)**
Generator (G):
Input: Random noise vector (latent space)
Fully connected layers + reshaping
Upsampling using convolutional layers
Activation:
ReLU (hidden layers)
Tanh (output layer)
Discriminator (D):
Input: Real or generated image
Convolutional layers for feature extraction
Dense layer with Sigmoid output
Activation: LeakyReLU (0.2)
Loss & Optimizer:
Binary Cross Entropy (BCE)
Adam Optimizer
Learning rate = 0.0002
Beta1 = 0.5
<img width="527" height="619" alt="image" src="https://github.com/user-attachments/assets/978287b0-8387-4275-92e9-beeebaf6b098" />


**Module 3 — Training & Monitoring**
This module handles GAN training logic.
Training Process:
Train Discriminator on:
Real images
Fake images
Train Generator to fool the Discriminator
Repeat for 100–300 epochs
Monitoring:
Track:
Generator loss (G_loss)
Discriminator loss (D_loss)
Save:
Generated image samples
Model checkpoints (G_final, D_final)
Outcome:
Images become sharper and more realistic as training progresses.

**Module 4 — Evaluation & Visualization**
Evaluates the quality and diversity of generated images.
Quantitative Metrics:
Classifier-based realism score
Diversity score (variation in outputs)
FID proxy score (real vs synthetic comparison)
Qualitative Analysis:
Visual inspection of generated samples
Mode collapse detection
Visualizations:
Loss curves (G_loss vs D_loss)
Image grids of generated samples
Latent space interpolation
t-SNE plots (real vs synthetic embeddings)

**Module 5 — Deployment Layer**
Makes the GAN usable in real applications.
Implemented Concepts:
Export trained Generator model
Inference script for batch image generation
Deployment options:
Streamlit UI
Flask / FastAPI API
Use Cases:
Generate N synthetic images on demand
Create artificial datasets for ML training
Control image size and randomness

**Module 6 — Monitoring & Update Pipeline**
Ensures long-term reliability and ethical use.
Monitoring:
Track inference latency
Monitor request frequency
Log generation failures
Model Updates:
Periodic retraining with new data
Version control:
G_v1, G_v2, G_v3
Privacy Assurance:
Ensure GAN does not memorize real samples
Compare generated images with nearest neighbors

**🏥 Real-Life Applications**
Healthcare: Synthetic X-ray / medical images
E-Commerce: Synthetic product images
Security: Synthetic CCTV frames
Manufacturing: Synthetic defect images
Education: Synthetic classroom / face datasets

**🛠️ Technologies Used**
Python
TensorFlow / Keras
NumPy, Matplotlib
Google Colab
Streamlit / Flask (deployment)

**▶️ How to Run**
Open the notebook in Google Colab
Run all cells sequentially
View generated images and loss curves
Use inference or deployment scripts for generation

**📌 Conclusion**
This project demonstrates that a Vanilla GAN can successfully generate privacy-preserving synthetic images. While effective for simple datasets, it also highlights the need for advanced GAN variants for complex, high-resolution images. The solution enables safe data sharing across multiple industries.
