import os

import torch
from pathlib import Path
import requests
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip


# Next we define our loss function, in this case the log-loss (or negative log-likelihood):
def nll(input, target):
    return (-input[range(target.shape[0]), target].log()).mean()


# Out model specification
def softmax(x):
    return x.exp() / x.exp().sum(-1).unsqueeze(-1)


# Below @ refers to matrix multiplication
def model(xb, weights, bias):
    return softmax(xb @ weights + bias)


# In the end, we are interested in the accuracy of our model
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def main():
    # First we download the dataset and unpackage it.
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    # PATH.mkdir(parents=True, exist_ok=True)

    # URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    # if not (PATH / FILENAME).exists():
    #    content = requests.get(URL + FILENAME).content
    #    (PATH / FILENAME).open("wb").write(content)

    # We then extract the data_depr and store it numpy arrays: x_train, y_train, x_valid, y_valid, x_test, y_test
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    # Check the shape of the x_valid holding the validation data_depr
    # print(x_valid.shape)

    # The images are stored in rows of length 784, hence to display the images we need to reshape them to 28×28.
    pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    # print(x_train.shape)

    # In order to take adavntage of PyTorch support for calculating gradients, we need to convert the numpy arrays to PyTorch tensors.
    # See the code example from the last lecture on PyTorch's support for automatic gradient calculation using the back propagation algorithm.
    x_train, y_train, x_valid, y_valid, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
    )
    n, c = x_train.shape
    print(f"n: {n}")
    x_train, x_train.shape, y_train.min(), y_train.max()
    # print(f"Training data_depr (images): \n {x_train}")
    # print(f"Training data_depr (labels): \n {y_train}")
    # print(f"Shape of x_train (now torch tensor) holding the training images: {x_train.shape}")
    # print(f"Min and max label values: {y_train.min()}, {y_train.max()}")

    # For the first part of this self study we will specify a neural network, which will encode a softmax function.
    # For this we need a (randomly initialized) weight matrix and a bias, and for both of them we need their gradients wrt.
    # our error function (yet to be defined) in order to perform learning.
    #
    # We store our weights (and biases) in a matrix structure so that the combination of inputs and weights can be expressed as one single matrix multiplication.
    weights = torch.randn(784, 10) / np.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)

    # Let's test our model (with our randomly initialized weights) using a so-called batch size of 64
    # (more on this later in the note book); for the prediction we pick out the first element in the batch.
    #
    # Note: During the lecture we didn't have time to cover learning with batches.
    # Before continuing, please revisit the slides/literature from lecture (in particular Slides 26 and 27)
    # and try to get the gist of this on your own. We will discuss it further at the next lecture on Thursday.
    batch_size = 64
    xb = x_train[0:batch_size]
    print(f"Batch shape: {xb.shape}")
    preds = model(xb, weights, bias)
    print(f"Prediction on first image {preds[0]}")
    print(f"Corresponding classification: {preds[0].argmax()}")

    loss_func = nll

    # Make a test calculation
    yb = y_train[0:batch_size]
    print(loss_func(preds, yb))

    print(f"Accuracy of model on batch (with random weights): {accuracy(preds, yb)}")

    # Now we are ready to combine it all and perform learning
    epochs = 10  # how many epochs to train for
    lr = 0.01  # learning rate
    regularization_balance = 0.03

    # We recorded the losses in lists for later plotting
    train_losses = []
    valid_losses = []

    # Iterate for a fixed number of epochs. One epoch is an iteration of the data_depr set, read in chucks of size batch_size
    for epoch in range(epochs):
        for batch_idx in range((n - 1) // batch_size + 1):

            # pick out the relevant batch
            start_i = batch_idx * batch_size
            end_i = start_i + batch_size
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]

            # Do prediction for all elements in the batch
            pred = model(xb, weights, bias)
            # and calculate the loss
            loss = loss_func(pred, yb)

            # Do back propagation to find the gradients
            loss.backward()
            with torch.no_grad():
                # Update the weights
                weights -= (1 - lr * regularization_balance) * weights.grad * lr
                #weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()

                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        train_loss = loss_func(model(x_train, weights, bias), y_train)
                        print(f"Epoch: {epoch}, B-idx: {batch_idx}, Training loss: {train_loss}")
                        train_losses.append(train_loss)

    # Plot the evolution of the training loss
    # plt.plot(range(len(train_losses)), train_losses, 'b')
    print(f"The accurracy is: {accuracy(model(xb, weights, bias), yb)}")

    # Exercise:
    #  1. Experiment with different variations of the gradient descent implementation;
    #  try varying the learning rate and the batch size. Assuming that you have a fixed time budget (say 2 minutes for learning),
    #  what can we then say about the effect of changing the parameters?

    #  2.  Implement momentum in the learning algorithm. How does it affect the results?
    #
    #  3. Try with different initialization schemes for the parameters (e.g. allowing for larger values).
    #  How does it affect the behavior of the algorithm?

    #  4. Analyze the behavior of the algorithm on the test set and implement a method for evaluating the accuracy
    #  over the entire training/test set (for inspiration, see Line 23 above)


if __name__ == "__main__":
    main()
