# Image Classification using CNN (Fashion-MNIST)

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify grayscale images of clothing items into 10 categories using the Fashion-MNIST dataset.

##  Overview

Automatically identifying the category of a clothing item from an image has real-world applications in e-commerce cataloging, inventory sorting, and visual search. This project trains a CNN to classify Fashion-MNIST images with high accuracy, avoiding the need for manual tagging.

##  Objective

Classify grayscale images of clothing items into 10 categories using a CNN, targeting **~92% test accuracy**.

##  Dataset

- **Name:** Fashion-MNIST
- **Size:** 60,000 training images + 10,000 test images
- **Image size:** 28x28 grayscale
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

##  Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Seaborn

##  Model Architecture

- Multiple Conv2D + BatchNormalization blocks
- MaxPooling2D for downsampling
- Dropout layers for regularization
- Dense layers with softmax output for 10-class classification

##  Workflow

1. Import libraries
2. Load and explore the dataset
3. Preprocess data (normalization, reshaping, one-hot encoding, train/validation split)
4. Apply data augmentation (rotation, shift, zoom)
5. Build the CNN model
6. Train the model (with `ReduceLROnPlateau` and `EarlyStopping` callbacks)
7. Plot training/validation accuracy & loss
8. Evaluate on the test set
9. Generate confusion matrix and classification report
10. Visualize sample predictions
11. Save the trained model

##  Results

- **Test Accuracy:** ~92%
- Confusion matrix and classification report included in the notebook, showing strong performance across most classes with minor confusion between visually similar categories (e.g., Shirt vs. T-shirt/top vs. Coat).

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Fashion_MNIST_CNN.git
   cd Fashion_MNIST_CNN
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn seaborn
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Image_Classification_Tiny_Project.ipynb
   ```

##  Project Structure

```
Fashion_MNIST_CNN/
├── Image_Classification_Tiny_Project.ipynb
├── fashion_mnist_cnn_model.h5      # Saved trained model
├── sample_images.png               # Sample dataset visualization
├── training_history.png            # Accuracy/loss plots
├── confusion_matrix.png            # Confusion matrix
├── sample_predictions.png          # Sample predictions
└── README.md
```

##  Conclusion

This CNN model was trained on the Fashion-MNIST dataset to classify clothing images into 10 categories. Using data augmentation, batch normalization, dropout regularization, and learning-rate scheduling, the model achieves approximately 92% test accuracy, demonstrating effective feature learning for image classification tasks.
