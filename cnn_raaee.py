import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
#load the mnist dataset
mnist= tf.keras.datasets.mnist
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
#Normalize pixel values between 0 and 1
X_train = X_train / 255.0 
x_test= x_test / 255.0
# Reshape data to include the channel dimension (28x28 grayscale -> 28x28x1)
x_train = X_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {x_test.shape}")

def create_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        # 1st Convolutional Layer: extracts 32 features using a 3x3 window
        layers.Conv2D(32, (3, 3), activation='relu', ),
        layers.MaxPooling2D((2, 2)),
        
        # 2nd Convolutional Layer: extracts 64 deeper features
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the 3D feature maps into a 1D vector for dense layers
        layers.Flatten(),
        
        # Fully Connected Layer
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),  # Prevents overfitting
        
        # Output Layer: 10 units for digits 0-9 with softmax probabilities
        layers.Dense(10, activation='softmax')
    ])
    return model

model = create_cnn_model()
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)
print("Final X Shape:", X_train.shape)  # MUST print exactly: (60000, 28, 28, 1)
print("Final Y Shape:", Y_train.shape)  # MUST print exactly: (60000,)


# Train the model

history = model.fit(X_train, Y_train, 
                    epochs=5, 
                    batch_size=64, 
                    validation_split=0.1)  # 10% data for validation

print("Training data shape:", x_train.shape) # Should be (60000, 28, 28, 1)
print("Testing data shape:", x_test.shape)   # Should be (10000, 28, 28, 1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# 5. SAMPLE PREDICTIONS VISUALIZATION

predictions = model.predict(x_test[:6])
predicted_labels = np.argmax(predictions, axis=1)

# Plot the sample results
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    color = 'green' if predicted_labels[i] == y_test[i] else 'red'
    plt.title(f"Act: {y_test[i]} | Pred: {predicted_labels[i]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()
