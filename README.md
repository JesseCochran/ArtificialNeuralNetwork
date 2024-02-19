# Artificial Neural Network

This Python program implements an Artificial Neural Network (ANN) for classification tasks. The ANN is built from scratch using numpy for mathematical operations and matplotlib for data visualization.

## How it works

The program starts by loading a dataset from a CSV file. The dataset is split into input (X) and output (y) data. The input data is then used to train the ANN, and the output data is used to evaluate the ANN's performance.

The ANN is configured with a specific number of input and output dimensions, a learning rate for gradient descent, and a regularization strength. These parameters can be adjusted in the `Config` class.

The ANN uses the tanh function as its activation function, and calculates loss using a custom loss function. The weights of the ANN are updated using gradient descent with weight regularization.

The program includes a function to visualize the decision boundary of the ANN. This function generates a grid of points and uses the trained ANN to predict the output for each point. The points are then colored according to their predicted output, creating a visual representation of the decision boundary.

## How to run the program

1. Ensure you have Python installed on your machine.
2. Clone this repository or download the `artificalNeuralNetwork.py` file.
3. Open a terminal/command prompt and navigate to the directory containing the `artificalNeuralNetwork.py` file.
4. Run the command `python artificalNeuralNetwork.py`.
5. The program will start and you can see the ANN in action.

## Example

```python
model = build_model(X, y, 3, print_loss=True, passes=10000)
In this example, the program builds an ANN with 3 nodes in the hidden layer. It trains the ANN for 10000 passes and prints the loss after every 1000 passes.
```

## Note
This is a simple program and does not use real-world images. The image is represented as a 2D array of numbers, where each number represents a pixel color. The flood fill algorithm is demonstrated step by step, with a 1-second pause between each step for visualization.

