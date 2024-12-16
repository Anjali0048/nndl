import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron model
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)  # Initialize random weights
        self.bias = np.random.rand(1)  # Initialize random bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        # Calculate weighted sum + bias, apply sigmoid activation
        z = np.dot(inputs, self.weights) + self.bias
        return sigmoid(z)

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for input_sample, target in zip(inputs, targets):
                # Forward pass
                prediction = self.predict(input_sample)

                # Calculate error
                error = target - prediction

                # Backpropagation: update weights and bias
                self.weights += self.learning_rate * error * sigmoid_derivative(prediction) * input_sample
                self.bias += self.learning_rate * error * sigmoid_derivative(prediction)

# Input data for AND and OR gates
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_targets = np.array([0, 0, 0, 1])  # AND gate outputs
or_targets = np.array([0, 1, 1, 1])   # OR gate outputs

# Train perceptron for AND gate
print("Training for AND gate...")
and_perceptron = Perceptron(input_size=2)
and_perceptron.train(inputs, and_targets, epochs=10000)

# Test AND gate
print("AND Gate Results:")
for input_sample in inputs:
    output = and_perceptron.predict(input_sample)
    print(f"Input: {input_sample}, Predicted Output: {round(output[0])}")

# Train perceptron for OR gate
print("\nTraining for OR gate...")
or_perceptron = Perceptron(input_size=2)
or_perceptron.train(inputs, or_targets, epochs=10000)

# Test OR gate
print("OR Gate Results:")
for input_sample in inputs:
    output = or_perceptron.predict(input_sample)
    print(f"Input: {input_sample}, Predicted Output: {round(output[0])}")
