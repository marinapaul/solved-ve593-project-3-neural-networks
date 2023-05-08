Download Link: https://assignmentchef.com/product/solved-ve593-project-3-neural-networks
<br>
<h1>Abstract</h1>

The goal of this project is to help you better understand Artificial Neural Networks and learn how to apply them to solve machine learning tasks.

<h1>Introduction</h1>

The project has two parts. In the first part, you need to implement artificial neural networks (ANNs) and some of the methods covered in class. Then in the second part, you will use your implementation to train an ANN to recognize handwritten digits.

<h1>Part 1: Neural Network</h1>

In this part, you will code a Python class for representing an ANN and its methods for inference, training and evaluation. In your implementation, you will use the numpy package, which is a library supporting large multidimensional arrays (in particular matrices) and providing fast operations on them. To learn more about this library, you can check <a href="https://www.numpy.org/">http://www.numpy. </a><a href="https://www.numpy.org/">org</a><a href="https://www.numpy.org/">.</a>

The file named networks.py, which we provide you, contains the partial definition of the class Network that will represent an ANN. Note that we assume that the ANN is a multi-layer perceptron for simplicity. To help you in this project, we provide you the constructor of Network, which we recall below:

def __init__(self, sizes): self.num_layers = len(sizes) self.sizes = sizes self.biases = [np.random.randn(y, 1) for y in sizes[1:]] self.weights = [np.random.randn(y, x)

for x, y in zip(sizes[:-1], sizes[1:])]

where the parameter sizes is a list containing the number of neurons in the respective layers of the network. For example, a 3-layer network with 10 neurons in the first layer, 5 neurons in the second layer, and 1 neuron in the last layer can be constructed with the following call: Network([10, 5, 1]).

Recall that in Python, the parameter self is not used during the invocation of a method. It provides a reference to the instance on which a method is called. Moreover, if a and b are two iterables (e.g., lists), zip(a, b) returns a sequence of pairs, where the first pair contains the first elements of a and b, the second pair the second elements of a and b and so on. The number of returned pairs depends on the length of the shorter iterable.

In this part, you will start by implementing the following methods:

<ul>

 <li>inference(self, x) which returns the output of the ANN for input x, which is assumed to be an 1-D array.</li>

 <li>training(self, trainData, T, n, alpha) which trains the ANN with training dataset trainData using stochastic gradient descent with mini-batches of size n, a total number of iteration T and learning rate alpha. This method calls the next one updateWeights at each iteration.</li>

 <li>updateWeights(self, batch, alpha), which updates the weights and biases of the ANN using the gradient (computed by the next method backprop) with mini-batch batch and learning rate alpha. The minibatch is represented as a list of pair (x, y).</li>

 <li>backprop(self, x, y) which returns a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y. The gradient nablaW and nablaB are layer-by-layer lists of arrays, following the same structure as weights and self.biases.</li>

 <li>evaluate(self, data) which returns the number of correct predictions of the current ANN on the (e.g., validation or test) dataset data, which is a list of pair (x, y). The prediction of the ANN is taken as the argmax of its output.</li>

</ul>

For the training part, you will first assume that the loss is the squared error and all the activation functions are sigmoids. To help you, we provide you the following functions:

<ul>

 <li>dSquaredLoss(a, y) returns the vector of partial derivatives of the squared loss with respect to the output activations for prediction a with correct label <em>y</em>, i.e. <em>a </em>− <em>y</em>.</li>

 <li>sigmoid(z) implements the sigmoid function.</li>

 <li>dSigmoid(z) implements the derivative of the sigmoid function.</li>

</ul>

When you have finished implementing the previous methods of Network, you should extend it so that an ANN could be trained with different activation functions in each layer, L2 regularization and/or early stopping:

<ul>

 <li>To allow different activation functions in each layer, a new parameter activationFcns, which is a layer-by-layer list of activation functions, can be added to the constructor. For instance, the previous 3-layer ANN would be now constructed by the following call:</li>

</ul>

Network([10, 5, 1], [None, sigmoid, sigmoid]).

Note that as the input layer does not use any activation function, we provide the value None. You will make sure that the correct activation functions are used for inference and training.

<ul>

 <li>To enable L2 regularization, you can simply add a new parameter lmbda (note lambda is a reserved word in Python), which corresponds to the <em>λ </em>hyperparameter of the regularization term, to the following methods training and updateWeights, and you should modify updateWeights accordingly to add the regularization term.</li>

 <li>To implement early-stopping, you can modify the method training and add a new parameter validationData to it. This new parameter represents a validation dataset that is used to track the performance of the ANN in order to decide when to stop the training. Note parameter <em>T </em>is still used, therefore the training can also be stopped if the number of iterations reaches <em>T</em>.</li>

</ul>

All new parameters should be added as last parameters of a method.

<h2>Coding Tasks</h2>

In networks.py, the interface for the previously mentioned functions are given. You need to provide the code for the methods with the missing implementation.

<strong>Important:</strong>

<ul>

 <li>You should implement the methods <strong>without changing the names and signatures</strong>.</li>

 <li>Frameworks including but not limited to pytorch, tensorFlow, theano are <strong>not allowed in this part</strong>.</li>

</ul>

Figure 1: Sample images from MNIST test dataset.

<h1>Part 2: MNIST</h1>

The MNIST (Modified National Institute of Standards and Technology database) database is a data set of images of handwritten digits. The images have been size-normalized and centered in a fixed-size image, see Figure 1.

The goal of this part is to train an ANN to learn to recognize these handwritten digits. This MNIST data set consists of many 28 by 28 pixel images of scanned handwritten digits. Therefore, the input layer of the ANN should contain 784 = 28 × 28 nodes. The input pixels are greyscale, with a value of 0<em>.</em>0 representing white, a value of 1<em>.</em>0 representing black, and inbetween values representing gradually darkening shades of grey.

The output layer of the ANN should contain 10 neurons to tell us which digit (0, 1, 2, 3, …, 9) corresponds to the input image. Intuitively, the <em>i</em>-th output node could be interpreted as how much the ANN believes that the input is digit <em>i</em>. The prediction of the ANN is assumed to be given by the argmax of its output.

<h2>Coding Tasks</h2>

For this part, we provide you the MNIST dataset (mnist.pkl.gz) and two helper functions (in database_loader.py), one for loading the dataset and the other for converting a digit (i.e., value from 0 to 9) to its one-hot representation. Note that the function that loads the data returns three data sets: training data, validation data and test data. The first data set is used for training (i.e., learning the weights and biases), the second may be used for choosing the ANN architecture or other hyperparameters and the last one is only used to evaluate your final trained model.

You will use your implementation of ANN realized in Part 1 to complete this task. In order to achieve the best performance possible, you will test different approaches (e.g., different activation functions, L2 regularization, early stopping) to train the ANN. In particular, following the examples of sigmoid and dsigmoid, you should implement at least these activation functions: tanh, and ReLU.

All the code necessary to reproduce your experiments should be saved in a file named experiments.py. For evaluation, you should output your training time, training accuracy and testing accuracy.

<strong>Important:</strong>

<ul>

 <li>You are free to choose the architecture of your ANN. We will provide bonus points if you compare different architectures. These experimental results should be explained in your report.</li>

 <li>You should explain how you chose the hyperparameters of your model</li>

</ul>

(e.g., <em>λ</em>, <em>n</em>…).

<ul>

 <li>You should only provide the code for and report the results of the experiments where the trained ANN has a testing accuracy <strong>above 90</strong>%</li>

 <li>Frameworks including but not limited to pytorch, tensorFlow, theano are <strong>not allowed in this part </strong>as well.</li>

</ul>


