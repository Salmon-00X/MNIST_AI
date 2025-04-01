import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten
import random
import sys

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

class CNNFeatureExtractor:
    def __init__(self):
        self.conv1 = Conv2D(32, (5, 5), activation='relu', padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu', padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()

    def extract_features(self, x, training=False):
        x = tf.expand_dims(x, axis=-1)  # Reshape (batch_size, 28, 28) -> (batch_size, 28, 28, 1)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        return self.flatten(x)

class MultiplyGate:
    def forward(self, W, X):
        return np.dot(X, W)

    def backward(self, W, X, dZ):
        dW = np.dot(X.T, dZ)
        dX = np.dot(dZ, W.T)
        return dW, dX

class AddGate:
    def forward(self, X, b):
        return X + b

    def backward(self, X, b, dZ):
        dX = dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        return db, dX

class ReLU:
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward(self, X, top_diff):
        return (X > 0) * top_diff

class Softmax:
    def predict(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))  # Improve stability
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        log_probs = -np.log(probs[range(num_examples), y] + 1e-8)  # Avoid log(0)
        return np.sum(log_probs) / num_examples

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs[range(num_examples), y] -= 1
        return probs / num_examples
    
class Model:
    def __init__(self, layers_dim):
        self.cnn = CNNFeatureExtractor()
        self.b = []
        self.W = []
        self.loss = []
        layers_dim = [1024, *layers_dim, 10]  # CNN output size (1024) -> Fully connected layers

        for i in range(len(layers_dim)-1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i+1]) * np.sqrt(2.0 / layers_dim[i]))
            self.b.append(np.zeros((1, layers_dim[i+1])))

    def calculate_loss(self, X, y):
        mulGate = MultiplyGate()
        addGate = AddGate()
        relu = ReLU()
        softmax = Softmax()

        input = self.cnn.extract_features(X).numpy()  # Convert to NumPy
        for i in range(len(self.W) - 1):
            input = relu.forward(addGate.forward(mulGate.forward(self.W[i], input), self.b[i]))
        input = addGate.forward(mulGate.forward(self.W[-1], input), self.b[-1])

        return softmax.loss(input, y)

    def predict(self, X):
        mulGate = MultiplyGate()
        addGate = AddGate()
        relu = ReLU()
        softmax = Softmax()

        input = self.cnn.extract_features(X).numpy()
        for i in range(len(self.W) - 1):
            input = relu.forward(addGate.forward(mulGate.forward(self.W[i], input), self.b[i]))
        input = addGate.forward(mulGate.forward(self.W[-1], input), self.b[-1])

        return np.argmax(softmax.predict(input), axis=1)

    def train(self, X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=False):
        mulGate = MultiplyGate()
        addGate = AddGate()
        relu = ReLU()
        softmax = Softmax()
        
        for epoch in range(num_passes):
            # Forward propagation
            input = self.cnn.extract_features(X).numpy()
            forward = [(None, None, input)]
            for i in range(len(self.W) - 1):
                mul = mulGate.forward(self.W[i], input)
                add = addGate.forward(mul, self.b[i])
                input = relu.forward(add)
                forward.append((mul, add, input))
            
            mul = mulGate.forward(self.W[-1], input)
            add = addGate.forward(mul, self.b[-1])
            output = softmax.predict(add)
            forward.append((mul, add, output))

            # Compute loss
            current_loss = softmax.loss(forward[-1][1], np.argmax(y, axis=1))
            self.loss.append(current_loss)
            
            # Backpropagation
            dZ = softmax.diff(forward[-1][1], np.argmax(y, axis=1))
            for i in range(len(forward) - 1, 0, -1):
                db, dmul = addGate.backward(forward[i][0], self.b[i-1], dZ)
                dW, dZ = mulGate.backward(self.W[i-1], forward[i-1][2], dmul)
                if i > 1:
                    dZ = relu.backward(forward[i-1][1], dZ)
                
                # Regularization
                dW += reg_lambda * self.W[i-1]
                
                # Update weights
                self.W[i-1] -= epsilon * dW
                self.b[i-1] -= epsilon * db
            
            if epoch%10 == 0:
                print(f"Epoch {epoch}: Loss = {current_loss:.4f}")

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_onehot = one_hot(y_train, 10)
y_test_onehot = one_hot(y_test, 10)

# Train model
layers_dim = [128, 64, 32]
model = Model(layers_dim)
model.train(X_train, y_train_onehot, num_passes=300, epsilon=0.1, reg_lambda=0.001, print_loss=True)

# Plot loss curve
plt.plot(model.loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

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
