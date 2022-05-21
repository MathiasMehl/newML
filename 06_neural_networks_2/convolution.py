import torch
from pathlib import Path
import requests
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
from scipy import signal

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

bw = np.array(x_train[0] > 0, dtype=int)
pyplot.imshow(bw.reshape((28, 28)), cmap="gray")

kernel = np.array([[1, -1]])
res = signal.convolve2d(bw.reshape((28, 28)), kernel, 'same')
pyplot.imshow(res.reshape((28, 28)), cmap="gray")

pyplot.show()

'''
row1 = [0, 0, 0, 0, 0, 0]
row2 = [0, 1, 0, 1, 1, 0]
row3 = [0, 1, 0, 1, 1, 0]
row4 = [0, 1, 0, 0, 0, 0]
row5 = [0, 1, 0, 0, 0, 0]
image = np.array([row1, row2, row3, row4, row5])
res = signal.convolve2d(image, kernel, 'same')
for row in res:
    print(row)
'''
