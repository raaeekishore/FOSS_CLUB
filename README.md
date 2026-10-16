# Handwritten Digit Classification using CNN

## Overview 

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify handwritten digits from 0 to 9. It uses the MNIST dataset, which contains grayscale images of handwritten digits.

The model learns patterns and features from the images and predicts the digit represented by each image.

## Features

- Uses the MNIST dataset for training and testing.
- Normalizes pixel values to the range `[0, 1]`.
- Implements a CNN with two convolutional layers and max-pooling layers.
- Uses a fully connected layer and dropout to help reduce overfitting.
- Evaluates the model on test data and reports test accuracy.
- Visualizes sample predictions, highlighting correct predictions in green and incorrect predictions in red.

## Model Architecture

The CNN consists of the following layers:

1. **Input Layer:** Accepts 28 × 28 grayscale images.
2. **Convolutional Layer 1:** Extracts 32 feature maps using 3 × 3 filters and ReLU activation.
3. **Max Pooling Layer 1:** Reduces the spatial dimensions of the feature maps.
4. **Convolutional Layer 2:** Extracts 64 deeper feature maps using 3 × 3 filters and ReLU activation.
5. **Max Pooling Layer 2:** Further reduces the spatial dimensions.
6. **Flatten Layer:** Converts the feature maps into a one-dimensional vector.
7. **Dense Layer:** Contains 64 neurons with ReLU activation.
8. **Dropout Layer:** Uses a dropout rate of 0.2 to help prevent overfitting.
9. **Output Layer:** Contains 10 neurons with softmax activation, representing the digits 0–9.

The model uses the Adam optimizer and sparse categorical cross-entropy loss. It is trained for 5 epochs with a batch size of 64, using 10% of the training data for validation.

## Dataset

The project uses the built-in MNIST dataset provided by TensorFlow.

- **Training images:** 60,000
- **Testing images:** 10,000
- **Image dimensions:** 28 × 28 pixels
- **Image type:** Grayscale
- **Classes:** 10 digits (0–9)

The pixel values are normalized by dividing them by 255.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Installation

Make sure Python and pip are installed on your system.

Install the required libraries using:

```bash
pip install tensorflow numpy matplotlib
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/raaeekishore/FOSS_CLUB.git
   ```

2. Navigate to the project directory:

   ```bash
   cd FOSS_CLUB
   ```

3. Run the Python script:

   ```bash
   python3 cnn_raaee.py
   ```

The script downloads the MNIST dataset if it is not already available locally, trains the CNN, evaluates its performance on the test dataset, and displays predictions for six sample images.

## Results

After training, the program prints the model summary, data shapes, and test accuracy.

It also displays six sample handwritten digits alongside their actual and predicted labels. Correct predictions are shown in green, while incorrect predictions are shown in red.

The actual test accuracy depends on the training run and environment.

## Future Improvements

- Experiment with different CNN architectures and hyperparameters.
- Train for more epochs and compare model performance.
- Visualize training and validation accuracy and loss.
- Test the model on handwritten digit images outside the MNIST dataset.

## Author

**Raaee Kishore**

GitHub: [@raaeekishore](https://github.com/raaeekishore)
