import tensorflow as tf
import numpy as np
from utils import load_data
import matplotlib.pyplot as plt
from circular_padding import CircularPadding2D

# Load your data
images, labels = load_data('data')

def train_test_split(images, labels, train_size=0.8):
    n = len(images)
    n_train = int(n * train_size)
    indices = np.random.permutation(n)
    train_indices, test_indices = indices[:n_train], indices[n_train:]
    return images[train_indices], labels[train_indices], images[test_indices], labels[test_indices]

# Split data into training and validation sets
X_train, y_train, X_val, y_val = train_test_split(images, labels)

# Placeholder for a model with Circular Padding (requires custom implementation)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16, 16, 1)),

    CircularPadding2D(padding=(3, 0)),  # Use the custom CircularPadding2D layer

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    CircularPadding2D(padding=(2, 2)),  # Use the custom CircularPadding2D layer again

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mse'])

# Callbacks
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit model
history = model.fit(X_train, y_train[:, 0], batch_size=10,
                    epochs=50,
                    validation_data=(X_val, y_val[:, 0]),
                    callbacks=[lr_schedule, early_stopping])

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time (Circular Padding)')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss = model.evaluate(X_val, y_val[:, 0])
print(f"Validation Loss with Circular Padding: {val_loss}")
