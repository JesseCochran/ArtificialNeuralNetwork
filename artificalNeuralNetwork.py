import numpy as np
import matplotlib.pyplot as plt
from csv import reader


def load_csv(filename):
    ###
    dataset = list()
    # Opens the file in read only mode

    with open(filename, 'r') as file:
        csv_reader = reader(file)

        for row in csv_reader:
            dataset.append(row)
###
    return dataset


def extract_only_x_data(dataset):
    ###
    data = list()

    for i in range(len(dataset)):
        data.append(list())

        for j in range(len(dataset[i]) - 1):
            data[-1].append(float(dataset[i][j]))
###
    return data


def extract_only_y_data(dataset):
    ###
    data = list()

    for i in range(len(dataset)):
        data.append(int(dataset[i][-1]))
###
    return data


class Config:
    # Specify input layer dimensionality,  output layer dimensionality, learning rate for gradient descent, regularization strength

    ###
    neural_network_input_dimension = 2
    neural_network_output_dimension = 2
    learning_rate = 0.01
    regularization_strength = 0.01
###


def generate_data():
    filename = 'moons.csv'

    # load the data from a csv file
    dataset = load_csv(filename)

    # extract the input data (2 inputs)
    x_data = extract_only_x_data(dataset)
    # extract the output data (1 output)
    y_data = extract_only_y_data(dataset)

    # convert the data into np array
    X = np.array(x_data)
    # convert the data into np array
    y = np.array(y_data)

    # returns both input (X) data and output (y) data
    return X, y


X, y = generate_data()

# Defines the size of the plot
plt.figure(figsize=(10, 7))
# Draws a scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def activation_function(x):
    ###
    # I tried to use sigmoid here however it resulted in a much greater loss unless i decreased the regularization_strength
    # sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
###
    # Return the value of the implemented activation function, do not simply return x
    return tanh


def loss(a):
    ###
    # Tried this loss function but resulted in a lot more loss
    # L = -np.log(1 - a)
    # I looked around online and managed to find a formula for the following which results in a lot less loss
    y = len(a)
    L = -1/y * np.sum(np.log(a))
###
    # Return the value of the implemented loss function, do not simply return a
    return L


def weight_regularization(Wx, dWx):
    ###
    regularisation = dWx + Config.regularization_strength * Wx
###
    # Return the value of the implemented weight regularization function, do not simply return dWx
    return regularisation


def forward_propagation(X, W1, b1, W2, b2):

    ###
    z1 = np.dot(X, W1) + b1
    a1 = activation_function(z1)

    z2 = np.dot(a1, W2) + b2
###
    return np.exp(z2)


def backward_propagation(X, y, W1, b1, W2, b2, exp_scores, number_of_examples):
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    delta3 = probs
    delta3[range(number_of_examples), y] -= 1

    z2 = np.dot(X, W1) + b1
    a1 = np.tanh(z2)

    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)

    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))

    dW1 = np.dot(X.T, delta2)

    # Add regularization terms (b1 and b2 don't have regularization terms)
    db1 = np.sum(delta2, axis=0)
###
    dW1 = dW1 + weight_regularization(W1, dW1)
    dW2 = dW2 + weight_regularization(W2, dW2)
###
    return dW1, dW2, db1, db2


def calculate_loss(model, X, y):

    number_of_examples = len(X)

    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

    exp_scores = forward_propagation(X, W1, b1, W2, b2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss

    correct_probs = loss(probs[range(number_of_examples), y])

    data_loss = np.sum(correct_probs)

    data_loss = data_loss + Config.regularization_strength / \
        2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1. / number_of_examples * data_loss


def predict(model, X):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

    exp_scores = forward_propagation(X, W1, b1, W2, b2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)


def build_model(X, y, number_of_nodes_within_hidden_layer, passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these

    number_of_examples = len(X)
    np.random.seed(0)

    # Two weights are needed as the network has two inputs
    # Likewise, two biases are needed as the network has two inputs

###
# YOUR CODE HERE
    W1 = np.random.randn(Config.neural_network_input_dimension,
                         number_of_nodes_within_hidden_layer) / np.sqrt(Config.neural_network_input_dimension)
    b1 = np.zeros((1, number_of_nodes_within_hidden_layer))
    W2 = np.random.randn(number_of_nodes_within_hidden_layer,
                         Config.neural_network_output_dimension) / np.sqrt(number_of_nodes_within_hidden_layer)
    b2 = np.zeros((1, Config.neural_network_output_dimension))
###

    model = {}

    for i in range(0, passes):
        # Forward Propgation
        exp_scores = forward_propagation(X, W1, b1, W2, b2)

        # Back Propgation
        dW1, dW2, db1, db2 = backward_propagation(
            X, y, W1, b1, W2, b2, exp_scores, number_of_examples)

        # Gradient descent parameter update

        ###

        W1 = W1 - dW1 * Config.learning_rate
        b1 = b1 - db1 * Config.learning_rate
        W2 = W2 - dW2 * Config.learning_rate
        b2 = b2 - db2 * Config.learning_rate
        ###

        # Assign new parameters to the model

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        result = calculate_loss(model, X, y)

        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, result))

            if result > 2.0:
                print("Loss is too high")
                break

    return model


model = build_model(X, y, 3, print_loss=True, passes=10000)


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.title("Artificial Neural Network")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

    def visualize(X, y, model):
        plot_decision_boundary(lambda x: predict(model, x), X, y)

    try:
        visualize(X, y, model)
    except:
        print("Can not visualize the graph, the data or the model is inconsistent")
