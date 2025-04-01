import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import random
import sys
import datetime

# Test GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

# Softmax , Cross-Entropy, Loss
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

class Convu:
    def __init__(self):
        # Define layers manually
        self.conv1 = Conv2D(32, (5, 5), activation=None, padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))
        
        self.conv2 = Conv2D(64, (5, 5), activation=None, padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2))

        self.flatten = Flatten()
        self.fc1 = Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = Dropout(0.5)
        self.fc2 = Dense(10, activation=None)

        # Custom loss function
        self.loss_fn = CustomSoftmaxCrossEntropy()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Initialize layers by passing a dummy input to build their variables
        dummy_input = tf.zeros((1, 28, 28, 1), dtype=tf.float32)
        self.call(dummy_input, training=True)

    def call(self, x, training=False):
        # Forward pass with manual activation (ReLU) where needed
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)  # No activation on final layer (logits)
        return x

    def train_step(self, X_batch, y_batch):
        # Collect trainable variables from all layers
        trainable_vars = (
            self.conv1.trainable_variables +
            self.bn1.trainable_variables +
            self.conv2.trainable_variables +
            self.bn2.trainable_variables +
            self.fc1.trainable_variables +
            self.fc2.trainable_variables
        )

        # Use GradientTape for gradient computation
        with tf.GradientTape() as tape:
            logits = self.call(X_batch, training=True)
            loss = self.loss_fn.compute_loss(logits, y_batch)

        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss, logits

    def get_trainable_variables(self):
        # Return all trainable variables for external use if needed
        return (
            self.conv1.trainable_variables +
            self.bn1.trainable_variables +
            self.conv2.trainable_variables +
            self.bn2.trainable_variables +
            self.fc1.trainable_variables +
            self.fc2.trainable_variables
        )

# Training function
def train_model(model, X_train, y_train, X_test, y_test_onehot, epochs=5, batch_size=64):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # TensorBoard setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
    test_summary_writer = tf.summary.create_file_writer(log_dir + '/test')
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        num_batches = len(X_train) // batch_size
        
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            loss, logits = model.train_step(X_batch, y_batch)

            predictions = tf.argmax(logits, axis=1)
            labels = tf.argmax(y_batch, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

            total_loss += loss
            total_accuracy += accuracy

            batch_progress = (batch_index + 1) / num_batches
            progress_bar = '=' * int(batch_progress * 50) + '>' + '.' * (50 - int(batch_progress * 50))
            print(f"Epoch {epoch + 1}/{epochs} [{batch_index + 1}/{num_batches}] [{progress_bar}] "
                  f"loss: {loss:.4f} - accuracy: {accuracy:.4f}", end='\r')

        print()

        avg_train_loss = total_loss / num_batches
        avg_train_accuracy = total_accuracy / num_batches

        # Consolidated batched evaluation for test set
        test_batch_size = 64
        total_test_loss = 0
        total_test_accuracy = 0
        num_test_samples = len(X_test)
        num_test_batches = 0

        for start in range(0, num_test_samples, test_batch_size):
            end = min(start + test_batch_size, num_test_samples)
            X_test_batch = X_test[start:end]
            y_test_batch = y_test_onehot[start:end]

            logits_test = model.call(X_test_batch, training=False)
            test_loss = model.loss_fn.compute_loss(logits_test, y_test_batch)
            predictions_test = tf.argmax(logits_test, axis=1)
            labels_test = tf.argmax(y_test_batch, axis=1)
            test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_test, labels_test), tf.float32))

            total_test_loss += test_loss
            total_test_accuracy += test_accuracy
            num_test_batches += 1

        avg_test_loss = total_test_loss / num_test_batches
        avg_test_accuracy = total_test_accuracy / num_test_batches

        train_losses.append(avg_train_loss.numpy())
        train_accuracies.append(avg_train_accuracy.numpy())
        test_losses.append(avg_test_loss.numpy())
        test_accuracies.append(avg_test_accuracy.numpy())

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_train_loss, step=epoch)
            tf.summary.scalar('accuracy', avg_train_accuracy, step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', avg_test_loss, step=epoch)
            tf.summary.scalar('accuracy', avg_test_accuracy, step=epoch)

        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss}, Accuracy: {avg_train_accuracy}, "
              f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")

    tf.keras.backend.clear_session()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in train_accuracies], label='Training Accuracy')
    plt.plot([acc * 100 for acc in test_accuracies], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_onehot = one_hot(y_train, 10)
y_test_onehot = one_hot(y_test, 10)

# Create model
model = Convu()

# Train the model
train_model(model, X_train, y_train_onehot, X_test, y_test_onehot, epochs=5, batch_size=64)

# Evaluate the model (also using batched prediction)
test_batch_size = 64
num_test_samples = len(X_test)
y_pred = []

for start in range(0, num_test_samples, test_batch_size):
    end = min(start + test_batch_size, num_test_samples)
    X_test_batch = X_test[start:end]
    logits_test = model.call(X_test_batch, training=False)
    predictions_test = tf.argmax(logits_test, axis=1)
    y_pred.extend(predictions_test.numpy())

y_pred = np.array(y_pred)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Display predictions on random test images
num_images = 5
fig, axes = plt.subplots(1, num_images, figsize=(10, 2))

for i in range(num_images):
    idx = random.randint(0, len(X_test) - 1)
    axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[i].set_title(f"Pred: {y_pred[idx]}")
    axes[i].axis("off")

plt.show()