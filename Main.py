
# -*- coding: utf-8 -*-
"""M22EE051.ipynb

# Network Architecture
"""

#necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# network architecture
input_dim =  784
output_dim = 10
hidden_layer1 = 128
hidden_layer2 = 64
hidden_layer3 = 32

#random seed based on roll number= M22EE051
seed = 51

#batch size as the year of admission
year_of_admission = 22
batch_size = year_of_admission
epochs = 25

#activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


#neural network
def build_model(input_dim, output_dim, hidden_layer1, hidden_layer2, hidden_layer3,seed):
    np.random.seed(seed)
    model = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_layer1': hidden_layer1,
        'hidden_layer2': hidden_layer2,
        'hidden_layer3': hidden_layer3,
        'weights': {
            'W1': np.random.randn(input_dim, hidden_layer1),
            'W2': np.random.randn(hidden_layer1, hidden_layer2),
            'W3': np.random.randn(hidden_layer2, hidden_layer3),
            'W4': np.random.randn(hidden_layer3, output_dim),
        },
        'biases': {
            'b1': np.ones((1, hidden_layer1)),
            'b2': np.ones((1, hidden_layer2)),
            'b3': np.ones((1, hidden_layer3)),
            'b4': np.ones((1, output_dim)),
        },
        'activations': {
            'sigmoid': sigmoid,
            'softmax': softmax,
        },
    }
    return model

"""# Implementation of feedforward propagation function"""

# Forward propagation function
def forward_propagation(X, model):
    A1 = np.dot(X, model['weights']['W1']) + model['biases']['b1'] #Pre-Activation of 1st layer
    H1 = model['activations']['sigmoid'](A1)  #Activation of 1st layer
    A2 = np.dot(H1, model['weights']['W2']) + model['biases']['b2'] #Pre-Activation of 2nd layer
    H2 = model['activations']['sigmoid'](A2) #Activation of 2nd layer
    A3 = np.dot(H2, model['weights']['W3']) + model['biases']['b3'] #Pre-Activation of 3rd layer
    H3 = model['activations']['sigmoid'](A3) #Activation of 3rd layer
    A4 = np.dot(H3, model['weights']['W4']) + model['biases']['b4'] #Pre-Activation of 4th layer
    H4 = model['activations']['softmax'](A4) #Activation of 4th layer=predicted output
    return A1, H1, A2, H2, A3, H3, A4, H4

"""# Implementation of Backpropagation and training function"""

# Backpropagation and training function
def train_model(X_train, y_train, input_dim, hidden_layer1, hidden_layer2, hidden_layer3, output_dim, epochs, batch_size, learning_rate,seed):
    model = build_model(input_dim, output_dim, hidden_layer1, hidden_layer2, hidden_layer3,seed)

    loss_history = []
    accuracy_history = []
    val_loss_history = []  # Store validation loss
    val_accuracy_history = []  # Store validation accuracy

    m = len(X_train)

    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            A1, H1, A2, H2, A3, H3, A4, H4 = forward_propagation(X_batch, model)

            dA4 =-(y_batch- H4)  #partial derivative of Cross entropy loss w.r.t. A4
            dW4 = np.dot(H3.T, dA4)
            db4 = np.sum(dA4, axis=0, keepdims=True)

            dH3 = np.dot(dA4, model['weights']['W4'].T) #partial derivative of Cross entropy loss w.r.t. H3
            dA3 = dH3 * H3 * (1 - H3)   #partial derivative of Cross entropy loss w.r.t. A3
            dW3 = np.dot(H2.T, dA3)
            db3 = np.sum(dA3, axis=0, keepdims=True)

            dH2 = np.dot(dA3, model['weights']['W3'].T)  #partial derivative of Cross entropy loss w.r.t. H2
            dA2 = dH2 * H2 * (1 - H2) #partial derivative of Cross entropy loss w.r.t. A2
            dW2 = np.dot(H1.T, dA2)
            db2 = np.sum(dA2, axis=0, keepdims=True)

            dH1 = np.dot(dA2, model['weights']['W2'].T) #partial derivative of Cross entropy loss w.r.t. H1
            dA1 = dH1 * H1 * (1 - H1) #partial derivative of Cross entropy loss w.r.t. A1
            dW1 = np.dot(X_batch.T, dA1)
            db1 = np.sum(dA1, axis=0, keepdims=True)

            # Update weights and biases
            model['weights']['W1'] -= learning_rate * dW1
            model['biases']['b1'] -= learning_rate * db1
            model['weights']['W2'] -= learning_rate * dW2
            model['biases']['b2'] -= learning_rate * db2
            model['weights']['W3'] -= learning_rate * dW3
            model['biases']['b3'] -= learning_rate * db3
            model['weights']['W4'] -= learning_rate * dW4
            model['biases']['b4'] -= learning_rate * db4

        #training loss and accuracy
        A1, H1, A2, H2, A3, H3, A4, H4 = forward_propagation(X_train, model)
        cost = -(1 / m) * np.sum(y_train * np.log(H4))
        loss_history.append(cost)

        predictions = np.argmax(H4, axis=1)
        accuracy = np.mean(predictions == np.argmax(y_train, axis=1))
        accuracy_history.append(accuracy)

        #validation loss and accuracy
        X_val = X_test
        y_val = y_test
        A1_val, H1_val, A2_val, H2_val, A3_val, H3_val, A4_val,H4_val = forward_propagation(X_val, model)
        val_cost = -(1 / len(X_val)) * np.sum(y_val * np.log(H4_val))
        val_loss_history.append(val_cost)

        val_predictions = np.argmax(H4_val, axis=1)
        val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
        val_accuracy_history.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, Accuracy: {accuracy * 100:.2f}%, Validation Loss: {val_cost:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%')

    return model, loss_history, accuracy_history, val_loss_history, val_accuracy_history

"""# Implementation of evaluation function"""

# Evaluation of model on the test set
def evaluate_model(X, model):
    A1, H1, A2, H2, A3, H3, A4, H4 = forward_propagation(X, model)
    return H4

"""# Implementation of confusion matrix function"""

#confusion matrix
def confusion_matrix_custom(y_true, y_pred, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    return conf_matrix

# Loading the dataset from data.csv
data = pd.read_csv('data.csv')

data.shape

"""# Implementaion of data spliting and one hot encoding function"""

# Data splitting function
def train_test_split_custom(data, test_size, random_state):
    np.random.seed(random_state)
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    split_index = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    return train_data, test_data


# One-hot encoding for the target labels
def to_categorical_custom(y, num_classes):
    return np.eye(num_classes)[y]

"""# Train-test splits in ratio of 90:10"""

# Split the data into training and testing sets
train_data, test_data = train_test_split_custom(data, test_size=0.1, random_state=seed)

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize the input data
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical_custom(y_train, output_dim)
y_test = to_categorical_custom(y_test, output_dim)

# Training the model
learning_rate = 0.01
trained_model, loss_history, accuracy_history, val_loss_history, val_accuracy_history = train_model(X_train, y_train, input_dim, hidden_layer1, hidden_layer2, hidden_layer3, output_dim, epochs, batch_size, learning_rate,seed)

# Plot accuracy and loss per epoch, including validation
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epoch')

plt.subplot(2, 2, 2)
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss/cost')
plt.legend()
plt.title('Loss vs. Epoch')

plt.show()

"""# Test accuracy for 90:10 train test split"""

A4_test = evaluate_model(X_test, trained_model)
predictions = np.argmax(A4_test, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix_custom(y_true, predictions, output_dim)

print('Test Accuracy: {:.2f}%'.format(np.mean(predictions == y_true) * 100))

"""# Confusion matrix for 90:10 train test split"""

print('Confusion Matrix:')
print(conf_matrix)

"""# Train-test splits in ratio of 80:20"""

# Split the data into training and testing sets
train_data, test_data = train_test_split_custom(data, test_size=0.2, random_state=seed)

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize the input data
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical_custom(y_train, output_dim)
y_test = to_categorical_custom(y_test, output_dim)

# Training the model
learning_rate = 0.01
trained_model, loss_history, accuracy_history, val_loss_history, val_accuracy_history = train_model(X_train, y_train, input_dim, hidden_layer1, hidden_layer2, hidden_layer3, output_dim, epochs, batch_size, learning_rate,seed)

# Plot accuracy and loss per epoch, including validation for
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epoch')

plt.subplot(2, 2, 2)
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss/cost')
plt.legend()
plt.title('Loss vs. Epoch')

plt.show()

"""# Test accuracy for 80:20 train test split"""

A4_test = evaluate_model(X_test, trained_model)
predictions = np.argmax(A4_test, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix_custom(y_true, predictions, output_dim)

print('Test Accuracy: {:.2f}%'.format(np.mean(predictions == y_true) * 100))

"""# Confusion matrix for 80:20 train test split"""

print('Confusion Matrix:')
print(conf_matrix)

"""# Train-test splits in ratio of 70:30"""

# Split the data into training and testing sets
train_data, test_data = train_test_split_custom(data, test_size=0.3, random_state=seed)

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize the input data
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical_custom(y_train, output_dim)
y_test = to_categorical_custom(y_test, output_dim)

# Training the model
learning_rate = 0.01
trained_model, loss_history, accuracy_history, val_loss_history, val_accuracy_history = train_model(X_train, y_train, input_dim, hidden_layer1, hidden_layer2, hidden_layer3, output_dim, epochs, batch_size, learning_rate,seed)

# Plot accuracy and loss per epoch, including validation
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epoch')

plt.subplot(2, 2, 2)
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss/cost')
plt.legend()
plt.title('Loss vs. Epoch')

plt.show()

"""# Test accuracy for 70:30 train test split"""

A4_test = evaluate_model(X_test, trained_model)
predictions = np.argmax(A4_test, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix_custom(y_true, predictions, output_dim)

print('Test Accuracy: {:.2f}%'.format(np.mean(predictions == y_true) * 100))

"""# Confusion matrix for 70:30 train test split"""

print('Confusion Matrix:')
print(conf_matrix)

"""# Total trainable and non-trainable parameters.

"""

# Report total trainable and non-trainable parameters
total_trainable_params = (input_dim * hidden_layer1 +
                          hidden_layer1 * hidden_layer2 +
                          hidden_layer2 * hidden_layer3 +
                          hidden_layer3 * output_dim +
                          hidden_layer1 +
                          hidden_layer2 +
                          hidden_layer3 +
                          output_dim)

total_non_trainable_params = input_dim + hidden_layer1 + hidden_layer2 + hidden_layer3 + output_dim+1 #learning rate

print(f'Total Trainable Parameters: {total_trainable_params}')
print(f'Total Non-Trainable Parameters: {total_non_trainable_params}')
