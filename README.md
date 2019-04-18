# VGGNet from Scratch
## Objectives
Implement VGG family from scratch, including MiniVGG and VGG16, and train them on CIFAR-10, and ImageNet datasets.
* Construct MiniVGGNet and train the network on CIFAR-10 datasets to obtain ≥85% accuracy.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 3.4.4
* [keras](https://keras.io/) 2.2.4
* [Tensorflow](https://www.tensorflow.org/install/) 1.12.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 9.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.1.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
### MiniVGG on CIFAR-10
The details about CIFAR-10 datasets can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

The MiniVGG can be found in `minivggnet.py` ([check here](https://github.com/meng1994412/VGGNet_from_scratch/blob/master/pipeline/nn/conv/minivggnet.py)) under `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, depth, and number of classes). In this part, the input would be (width = 32, height = 32, depth = 3, classes = 10).

Table 1 shows the MiniVGG architecture. The activation and batch normalization layer is not shown in the table, which should be after each `CONV` layer and `FC` layer. The `ReLU` activation function is used in the project.

Table 1: MiniVGG architecture.

| Layer Type    | Output Size     | Filter Size / Stride    |
| ------------- |:---------------:| -----------------------:|
| Input Image   | 32 x 32 x 3     |                         |
| CONV          | 32 x 32 x 32    | 3 x 3, K = 96           |
| CONV          | 32 x 32 x 32    | 3 x 3, K = 32           |
| POOL          | 16 x 16 x 32    | 2 x 2                   |
| DROPOUT       | 16 x 16 x 32    |                         |
| CONV          | 16 x 16 x 64    | 3 x 3, K = 64           |
| CONV          | 16 x 16 x 64    | 3 x 3, K = 64           |
| POOL          | 8 x 8 x 64      | 2 x 2                   |
| DROPOUT       | 8 x 8 x 64      |                         |
| FC            | 512             |                         |
| DROPOUT       | 512             |                         |
| FC            | 10              |                         |
| softmax       | 10              |                         |

The `minivggnet_cifar10.py` ([check here](https://github.com/meng1994412/VGGNet_from_scratch/blob/master/minivggnet_cifar10.py)) is responsible for training the network, evaluating the model (including plotting the loss and accuracy curve of training and validation sets, providing the classification report), and serialize the model to disk.

## Results
### MiniVGG on CIFAR-10
Figure 1 demonstrates the loss and accuracy curve of training and validation sets. And Figure 2 shows the evaluation of the network, which indicate a 86% accuracy. During this training process, Stochastic Gradient Descent with momentum and Nesterov acceleration is used.

<img src="https://github.com/meng1994412/VGGNet_from_scratch/blob/master/output/8809.png" width="500">

Figure 1: Plot of training and validation loss and accuracy (SGD).

<img src="https://github.com/meng1994412/VGGNet_from_scratch/blob/master/output/minivggnet_cifar10_evaluation_2.png" width="400">

Figure 2: Evaluation of the network, indicating 86% accuracy (SGD).

After evaluating the training process, I change the training optimizer from `SGD` to `Adam` to check whether `Adam` can boost the accuracy a little bit, since `Adam` can converge faster than `SGD`.

Figure 3 demonstrates the loss and accuracy curve of training and validation sets. And Figure 4 shows the evaluation of the network, which indicate a 88% accuracy. As we can see, the accuracy is boosted to 88% with about 2% increment due to the `Adam` optimizer.

<img src="https://github.com/meng1994412/VGGNet_from_scratch/blob/master/output/9474.png" width="500">

Figure 3: Plot of training and validation loss and accuracy (Adam).

<img src="https://github.com/meng1994412/VGGNet_from_scratch/blob/master/output/minivggnet_cifar10_evaluation_3.png" width="400">

Figure 4: Evaluation of the network, indicating 88% accuracy (Adam).
