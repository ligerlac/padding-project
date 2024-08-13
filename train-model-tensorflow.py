import tensorflow as tf
import numpy as np
from utils import load_data
import matplotlib.pyplot as plt

# Load your data (assuming you have the same function to load data)
images, labels = load_data('data')


# Split data into training and validation sets
def train_test_split(images, labels, train_size=0.8):
    n = len(images)
    n_train = int(n * train_size)
    indices = np.random.permutation(n)
    train_indices, test_indices = indices[:n_train], indices[n_train:]
    return images[train_indices], labels[train_indices], images[test_indices], labels[test_indices]


X_train, y_train, X_val, y_val = train_test_split(images, labels)

# Normalize the data (if necessary)
X_train = X_train / 255.0
X_val = X_val / 255.0

# Define a Sequential model
model = tf.keras.Sequential([
    # Convolutional Layer 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Layer 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the output for the Dense layers
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(64, activation='relu'),

    # Output Layer
    tf.keras.layers.Dense(1, activation='linear')  # Output is a single value, e.g., the width 'w'
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_squared_error',
              metrics=['mse'])

# Train the model
history = model.fit(X_train, y_train[:, 0],
                    epochs=10, batch_size=10,
                    validation_data=(X_val, y_val[:, 0]))

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss = model.evaluate(X_val, y_val[:, 0])
print(f"Validation Loss: {val_loss}")