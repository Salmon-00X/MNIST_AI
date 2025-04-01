import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import random

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# One-hot encoding
def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

class CustomSoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None

    def softmax(self, logits):
        exp_logits = tf.exp(logits - tf.reduce_max(logits, axis=1, keepdims=True))
        return exp_logits / tf.reduce_sum(exp_logits, axis=1, keepdims=True)

    def compute_loss(self, logits, labels):
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        
        one_hot_labels = tf.one_hot(labels, logits.shape[1])
        self.probs = self.softmax(logits)
        loss = -tf.reduce_mean(tf.reduce_sum(one_hot_labels * tf.math.log(self.probs + 1e-8), axis=1))
        return loss

    def gradient(self, logits, labels):
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        
        one_hot_labels = tf.one_hot(labels, logits.shape[1])
        grad = self.probs - one_hot_labels
        return grad

class CNNFeatureExtractor:
    def __init__(self):
        self.conv1 = Conv2D(32, (5, 5), activation=None, padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))
        
        self.conv2 = Conv2D(64, (5, 5), activation=None, padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2))
        
        self.flatten = Flatten()

    def extract_features(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        return x

class Model:
    def __init__(self, layers_dim):
        self.cnn = CNNFeatureExtractor()
        self.loss_fn = CustomSoftmaxCrossEntropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = []

        layers_dim = [1024, *layers_dim, 10]  # CNN output size (1024) -> Fully connected layers
        self.W = []
        self.b = []
        
        # Initialize weights and biases for fully connected layers
        for i in range(len(layers_dim)-1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i+1]) * np.sqrt(2.0 / layers_dim[i]))
            self.b.append(np.zeros((1, layers_dim[i+1])))

    def relu(self, X):
        return np.maximum(0, X)
    
    def softmax(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))  # Improve stability
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def softmax_loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.softmax(X)
        log_probs = -np.log(probs[range(num_examples), y] + 1e-8)  # Avoid log(0)
        return np.sum(log_probs) / num_examples
    
    def softmax_diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.softmax(X)
        probs[range(num_examples), y] -= 1
        return probs / num_examples

    def calculate_loss(self, X, y):
        input = self.cnn.extract_features(X)
        for i in range(len(self.W) - 1):
            input = self.relu(np.dot(input, self.W[i]) + self.b[i])
        input = np.dot(input, self.W[-1]) + self.b[-1]
        return self.softmax_loss(input, y)

    def predict(self, X):
        input = self.cnn.extract_features(X)
        for i in range(len(self.W) - 1):
            input = self.relu(np.dot(input, self.W[i]) + self.b[i])
        input = np.dot(input, self.W[-1]) + self.b[-1]
        return np.argmax(self.softmax(input), axis=1)

    def train(self, X, y, num_epochs=10, batch_size=64, epsilon=0.01, reg_lambda=0.01):
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataset:
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)
                
                input = self.cnn.extract_features(X_batch)
                forward = [(None, None, input)]
                
                for i in range(len(self.W) - 1):
                    input = self.relu(np.dot(input, self.W[i]) + self.b[i])
                    forward.append((self.W[i], self.b[i], input))
                
                final_scores = np.dot(input, self.W[-1]) + self.b[-1]
                forward.append((self.W[-1], self.b[-1], final_scores))
                
                current_loss = self.softmax_loss(final_scores, np.argmax(y_batch, axis=1))
                epoch_loss += current_loss
                
                dZ = self.softmax_diff(final_scores, np.argmax(y_batch, axis=1))
                for i in range(len(forward) - 1, 0, -1):
                    dW = tf.matmul(tf.transpose(forward[i-1][2]), dZ) + reg_lambda * self.W[i-1]
                    db = np.sum(dZ, axis=0, keepdims=True)
                    
                    if i > 1:
                        dZ = np.dot(dZ, tf.transpose(self.W[i-1])) * (forward[i-1][2] > 0)
                    
                    self.W[i-1] -= epsilon * dW
                    self.b[i-1] -= epsilon * db
            
            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
y_train_onehot = one_hot(y_train, 10)
y_test_onehot = one_hot(y_test, 10)

# Train model
model = Model(layers_dim=[128, 64, 32])
model.train(X_train, y_train_onehot, num_epochs=10, batch_size=64)

# Plot loss
plt.plot(model.loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

# Evaluate model
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Show random images and predicted labels
num_images = 5
fig, axes = plt.subplots(1, num_images, figsize=(num_images*2, 2))

for i in range(num_images):
    idx = random.randint(0, len(X_test) - 1)
    axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[i].set_title(f"Pred: {y_pred[idx]}")
    axes[i].axis("off")

plt.show()