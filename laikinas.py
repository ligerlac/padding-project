import numpy as np
import matplotlib.pyplot as plt

# Define edge detection kernels
vertical_kernel = np.array([[ 1,  0, -1],
                            [ 1,  0, -1],
                            [ 1,  0, -1]])

horizontal_kernel = np.array([[ 1,  1,  1],
                              [ 0,  0,  0],
                              [-1, -1, -1]])

# asfdasfas

# loading data
def load_data(path):
    with open(f'{path}/images.npy', 'rb') as f:
        images = np.load(f)
    with open(f'{path}/labels.npy', 'rb') as f:
        labels = np.load(f)
    return images, labels

# load data
images, labels = load_data('data')

# split data
def train_test_split(images, labels, train_size=0.8):
    n = len(images)
    n_train = int(n * train_size)
    indices = np.random.permutation(n)
    train_indices, test_indices = indices[:n_train], indices[n_train:]
    return images[train_indices], labels[train_indices], images[test_indices], labels[test_indices]

X_train, y_train, X_val, y_val = train_test_split(images, labels)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))

# ---------------------

# Define the convolution operation
def conv2d(image, kernel, stride=1, padding=0):
    # Add zero-padding to the input image
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Get the dimensions of the image and the kernel
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate the dimensions of the output feature map
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    # Initialize the output feature map
    output = np.zeros((output_h, output_w))

    # Perform the convolution operation
    for i in range(0, output_h):
        for j in range(0, output_w):
            region = image[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

# Max Pooling Layer
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



# Define a ReLU activation function
def relu(Z):
    return np.maximum(0, Z)

# Convolutional Layer Forward Propagation
def conv_layer_forward(X, kernel, stride=1, padding=0):
    return relu(conv2d(X, kernel, stride, padding))

# Define a function for the fully connected layer
def fully_connected_layer(input_vector, weights, biases, activation_function=None):
    """
    Performs a fully connected layer operation.

    Parameters:
    - input_vector: The input to the layer (flattened feature vector)
    - weights: The weights of the layer
    - biases: The biases of the layer
    - activation_function: Optional activation function (e.g., ReLU, sigmoid)

    Returns:
    - output_vector: The output of the layer after applying the activation function
    """
    # Compute the linear transformation
    linear_output = np.dot(input_vector, weights) + biases

    # Apply activation function if provided
    # Sent the function a name for the activasion function
    if activation_function is not None:
        return activation_function(linear_output)
    else:
        return linear_output


# Define a simple neural network with one hidden fully connected layer
def neural_network(X, weights1, biases1, weights2, biases2):
    """
    Perform a forward pass through a simple neural network with one hidden layer.

    Parameters:
    - X: Input data
    - weights1, biases1: Weights and biases for the hidden layer
    - weights2, biases2: Weights and biases for the output layer

    Returns:
    - output: Output of the neural network
    """
    # Forward pass through the hidden layer
    hidden_layer_output = fully_connected_layer(X, weights1, biases1, activation_function=relu)

    # Forward pass through the output layer
    output = fully_connected_layer(hidden_layer_output, weights2, biases2, activation_function=relu)

    return output

# Define the loss function and its derivative
def binary_cross_entropy_loss(predictions, labels):
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    loss = -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
    return loss


def compute_gradients(X, y, weights1, biases1, weights2, biases2):
    hidden_layer_output = fully_connected_layer(X, weights1, biases1, activation_function=relu)
    predictions = fully_connected_layer(hidden_layer_output, weights2, biases2, activation_function=relu)

    loss = binary_cross_entropy_loss(predictions, y)
    dL_dpred = (predictions - y) / (predictions * (1 - predictions))  # Gradient of loss w.r.t predictions

    dL_dhidden = np.dot(dL_dpred, weights2.T) * (hidden_layer_output > 0)  # ReLU derivative

    grads = {
        'dW2': np.dot(hidden_layer_output.T, dL_dpred),
        'db2': np.sum(dL_dpred, axis=0),
        'dW1': np.dot(X.T, dL_dhidden),
        'db1': np.sum(dL_dhidden, axis=0)
    }

    return grads, loss

def update_parameters(weights1, biases1, weights2, biases2, grads, learning_rate):
    weights1 -= learning_rate * grads['dW1']
    biases1 -= learning_rate * grads['db1']
    weights2 -= learning_rate * grads['dW2']
    biases2 -= learning_rate * grads['db2']


# Training loop
def train(X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size):
    input_size = X_train.shape[1] * X_train.shape[2]  # Flattened image size
    hidden_layer_size = 10
    output_size = 1

    weights1 = np.random.randn(input_size, hidden_layer_size) * 0.01
    biases1 = np.zeros(hidden_layer_size)
    weights2 = np.random.randn(hidden_layer_size, output_size) * 0.01
    biases2 = np.zeros(output_size)

    num_train_samples = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0

        # Shuffle training data
        permutation = np.random.permutation(num_train_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        # Mini-batch gradient descent
        for start in range(0, num_train_samples, batch_size):
            end = min(start + batch_size, num_train_samples)
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Apply convolution and pooling to each image in the batch
            batch_features = []
            for img in X_batch:
                vertical = conv_layer_forward(img, vertical_kernel)
                horizontal = conv_layer_forward(img, horizontal_kernel)
                pooled_vertical = max_pooling(vertical)
                pooled_horizontal = max_pooling(horizontal)
                combined_features = np.concatenate(
                    (pooled_vertical[..., np.newaxis], pooled_horizontal[..., np.newaxis]), axis=-1)
                flattened_features = combined_features.flatten()
                batch_features.append(flattened_features)

            batch_features = np.array(batch_features)

            grads, loss = compute_gradients(batch_features, y_batch, weights1, biases1, weights2, biases2)
            update_parameters(weights1, biases1, weights2, biases2, grads, learning_rate)
            epoch_loss += loss * X_batch.shape[0]

        epoch_loss /= num_train_samples

        if epoch % 10 == 0:
            train_predictions = neural_network(X_train, weights1, biases1, weights2, biases2)
            train_loss = binary_cross_entropy_loss(train_predictions, y_train)
            val_predictions = neural_network(X_val, weights1, biases1, weights2, biases2)
            val_loss = binary_cross_entropy_loss(val_predictions, y_val)
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Example usage
    epochs = 100
    learning_rate = 0.01
    batch_size = 10
    train(X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size)