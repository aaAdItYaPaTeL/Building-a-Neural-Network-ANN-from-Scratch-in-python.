# Building-a-Neural-Network-ANN-from-Scratch-in-python-

Objective:
In this assignment, you are required to implement a neural network from scratch in Python.
Build a feedforward neural network and implement backpropagation for training. By the end of
this assignment, you should have a working neural network that can be trained on a simple
dataset for multi-class classification.
Dataset:
Get the dataset from here. The dataset consists of 70k images representing 10 different object
categories.
Network Architecture:
1. The network should contain 3 hidden layers. Excluding input and output layers. Set the
network architecture as:
a. Input layer = set to the size of the dimensions
b. Output layer = set to the size of the #classes
c. Hidden layer1 = 128
d. Hidden layer2 = 64
e. Hidden layer3 = 32
2. Initialize the weights randomly using seed value as the last three digits of your roll
number. For example, your roll number is P23CS001, then your seed value should be 1.
Set bias = 1.
3. Use Train-test splits as randomized 70:30, 80:20 and 90:10.
4. Set batch size as your year of admission. For example, your roll number is P23CS001,
then your batch size should be 23.
5. Set ‘sigmoid’ as activation function for hidden layers and ‘softmax’ for output layer.
6. Use Gradient Descent for optimization. Loss function as crossentropy.
7. Train for 25 epochs. Plot accuracy and loss per epoch.
8. Prepare a Confusion matrix for all the combinations of the network. You may use an
in-built function for this purpose.
9. Report total trainable and non-trainable parameters.
