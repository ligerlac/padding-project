import numpy as np
import matplotlib.pyplot as plt
from utils import load_data

# Load data
images, labels = load_data('data')

# Split data
def train_test_split(images, labels, train_size=0.8):
    n = len(images)
    n_train = int(n * train_size)
    indices = np.random.permutation(n)
    train_indices, test_indices = indices[:n_train], indices[n_train:]
    return images[train_indices], labels[train_indices], images[test_indices], labels[test_indices]

X_train, y_train, X_val, y_val = train_test_split(images, labels)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))

# Define max pooling function
def max_pooling(image, pool_size=2, stride=2):
    image_h, image_w = image.shape
    output_h = (image_h - pool_size) // stride + 1
    output_w = (image_w - pool_size) // stride + 1

    output = np.zeros((output_h, output_w))

    for i in range(0, output_h):
        for j in range(0, output_w):
            region = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)

    return output

# Define convolution and ReLU functions
def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w))

    for i in range(0, output_h):
        for j in range(0, output_w):
            region = image[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

def relu(Z):
    return np.maximum(0, Z)

def conv_layer_forward(X, kernel, stride=1, padding=0):
    Z = conv2d(X, kernel, stride, padding)
    A = relu(Z)
    return A

def initialize_fc_layer(input_size, output_size):
    # Randomly initialize weights and biases
    np.random.seed(42)
    W = np.random.randn(input_size, output_size) * 0.01
    b = np.zeros((1, output_size))
    return W, b

def fc_forward(X, W, b):
    # Fully connected layer forward pass
    Z = np.dot(X, W) + b
    return Z

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


def forward_pass(X_batch, W_conv, b_conv, W_fc, b_fc):
    batch_size = X_batch.shape[0]
    flattened_batch = []

    for i in range(batch_size):
        vertical_output = conv_layer_forward(X_batch[i], W_conv, stride=1, padding=0)
        horizontal_output = conv_layer_forward(X_batch[i], W_conv.T, stride=1, padding=0)

        pooled_vertical = max_pooling(vertical_output, pool_size=2, stride=2)
        pooled_horizontal = max_pooling(horizontal_output, pool_size=2, stride=2)

        combined_features = np.concatenate((pooled_vertical.flatten(), pooled_horizontal.flatten()))
        flattened_batch.append(combined_features)

    flattened_batch = np.array(flattened_batch)
    fc_output = fc_forward(flattened_batch, W_fc, b_fc)

    return flattened_batch, fc_output


def backward_pass(flattened_batch, y_batch, fc_output, W_conv, b_conv, W_fc, b_fc, learning_rate):
    # Calculate loss and its derivative
    loss = mean_squared_error(y_batch, fc_output)
    loss_grad = mse_loss_derivative(y_batch, fc_output)

    # Gradients for fully connected layer
    dW_fc = np.dot(flattened_batch.T, loss_grad)
    db_fc = np.sum(loss_grad, axis=0, keepdims=True)

    # print("Shape of W_fc:", W_fc.shape)
    # print("Shape of dW_fc:", dW_fc.shape)
    # print("Shape of X_batch:", flattened_batch.shape)

    # Update fully connected layer weights and biases
    W_fc -= learning_rate * dW_fc
    b_fc -= learning_rate * db_fc

    return W_conv, b_conv, W_fc, b_fc, loss

def evaluate_model(X_val, y_val, W_conv, b_conv, W_fc, b_fc):
    flattened_batch, fc_output = forward_pass(X_val, W_conv, b_conv, W_fc, b_fc)
    loss = mean_squared_error(y_val, fc_output)
    return loss

def train_network(X_train, y_train, X_val, y_val, W_conv, b_conv, W_fc, b_fc, epochs, batch_size, learning_rate):
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        batch_losses = []
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            # Forward pass
            flattened_batch, fc_output = forward_pass(X_batch, W_conv, b_conv, W_fc, b_fc)

            # Backward pass and update
            W_conv, b_conv, W_fc, b_fc, loss = backward_pass(flattened_batch, y_batch, fc_output, W_conv, b_conv, W_fc, b_fc,
                                                             learning_rate)

            batch_losses.append(loss)

        print(f"--- Epoch {epoch + 1}/{epochs}")

        # Print average loss for the epoch
        avg_loss = np.mean(batch_losses)
        train_loss_history.append(avg_loss)
        print(f"Loss: {avg_loss:.4f}")

        # Evaluate on validation data
        val_loss = evaluate_model(X_val, y_val, W_conv, b_conv, W_fc, b_fc)
        val_loss_history.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

    return W_conv, b_conv, W_fc, b_fc, train_loss_history, val_loss_history

# Determine the input size for the fully connected layer using a sample image
def calculate_input_size(X_sample, W_conv):
    vertical_output = conv_layer_forward(X_sample, W_conv, stride=1, padding=0)
    horizontal_output = conv_layer_forward(X_sample, W_conv.T, stride=1, padding=0)

    pooled_vertical = max_pooling(vertical_output, pool_size=2, stride=2)
    pooled_horizontal = max_pooling(horizontal_output, pool_size=2, stride=2)

    combined_features = np.concatenate((pooled_vertical.flatten(), pooled_horizontal.flatten()))

    return combined_features.shape[0]


def plot_layer_outputs(X_sample, W_conv):
    vertical_output = conv_layer_forward(X_sample, W_conv, stride=1, padding=0)
    horizontal_output = conv_layer_forward(X_sample, W_conv.T, stride=1, padding=0)

    pooled_vertical = max_pooling(vertical_output, pool_size=2, stride=2)
    pooled_horizontal = max_pooling(horizontal_output, pool_size=2, stride=2)

    plt.figure(figsize=(12, 8))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(X_sample, cmap='gray')

    # Vertical Convolution Output
    plt.subplot(2, 3, 2)
    plt.title("Vertical Convolution Output")
    plt.imshow(vertical_output, cmap='gray')

    # Horizontal Convolution Output
    plt.subplot(2, 3, 3)
    plt.title("Horizontal Convolution Output")
    plt.imshow(horizontal_output, cmap='gray')

    # Pooled Vertical Output
    plt.subplot(2, 3, 4)
    plt.title("Pooled Vertical Output")
    plt.imshow(pooled_vertical, cmap='gray')

    # Pooled Horizontal Output
    plt.subplot(2, 3, 5)
    plt.title("Pooled Horizontal Output")
    plt.imshow(pooled_horizontal, cmap='gray')

    plt.tight_layout()
    plt.show()


# Load and prepare data
images, labels = load_data('data')

# Define edge detection kernels
vertical_kernel = np.array([[ 1,  0, -1],
                            [ 1,  0, -1],
                            [ 1,  0, -1]])

horizontal_kernel = np.array([[ 1,  1,  1],
                              [ 0,  0,  0],
                              [-1, -1, -1]])

# Sample image from the training set
X_sample = X_train[0]

plot_layer_outputs(X_sample, vertical_kernel)

# Calculate the input size
input_size = calculate_input_size(X_sample, vertical_kernel)
print("Input Size for Fully Connected Layer:", input_size)

# Initialize the fully connected layer with the correct input size
W_fc, b_fc = initialize_fc_layer(input_size=input_size, output_size=1)
W_conv = vertical_kernel  # Example: using your vertical kernel for simplicity
b_conv = np.zeros((1, 1))  # Example bias for convolutional layer

# Train the network
epochs = 10
batch_size = 10
learning_rate = 0.01

W_conv, b_conv, W_fc, b_fc, train_loss_history, val_loss_history = train_network(
    X_train, y_train[:, 0].reshape(-1, 1), X_val, y_val[:, 0].reshape(-1, 1),
    W_conv, b_conv, W_fc, b_fc, epochs, batch_size, learning_rate)

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.show()
