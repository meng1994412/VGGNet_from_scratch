# set matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pipeline.nn.conv import MiniVGGNet
from pipeline.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# define the total number of epochs to train for along with initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to output model")
ap.add_argument("-o", "--output", required = True,
    help = "path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# load the training and testing data, converting the image from integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis = 0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label name for CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1,
    horizontal_flip = True, fill_mode = "nearest")

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath = jsonPath),
    LearningRateScheduler(poly_decay)]

# initialize the optimizer and model
print("[INFO] compiling model...")
# opt = SGD(lr = INIT_LR, momentum = 0.9, nesterov = True)
opt = Adam(lr = INIT_LR)
model = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size = 64),
    validation_data = (testX, testY), steps_per_epoch = len(trainX) // 64,
    epochs = NUM_EPOCHS, callbacks = callbacks, verbose = 1)

# evaluate network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1),
    predictions.argmax(axis = 1), target_names = labelNames))

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])
