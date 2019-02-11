"""Tensorflow fizz buzz
"""
import numpy as np
import tensorflow as tf

NUM_DIGITS = 24 # number of dimensions for the binary encoding
N_MAX = 4096 # maximum integer to be used for training our neural net
NUM_HIDDEN = 200 # number of nodes in the hidden layer
NUM_EPOCHS = 2000 # number of epochs to train
BATCH_SIZE = 250 # batch size for training

def generate_fizz_buzz_sequence(n_max):
    """Function to generate fizz buzz sequence
    """
    output_list = []
    for number in range(1, n_max + 1):
        if number % 15 == 0:
            output_list.append("fizzbuzz")
        elif number % 3 == 0:
            output_list.append("fizz")
        elif number % 5 == 0:
            output_list.append("buzz")
        else:
            output_list.append(str(number))

    return output_list

def binary_encode(i, num_digits):
    """Function to encode integers into binary "vectors"

    Parameters
    ----------
    i: int
        Integer to encode
    num_digits: int
        Dimension of the binary "vector"

    Returns
    -------
    np.array
        Array holding the binary vector representation of the the integer i
    """
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    """Function to encode fizz buzz target

    This function encodes an integer into a 4d vector that represents our fizz buzz target of
    either 'integer', 'fizz', 'buzz' and 'fizzbuzz'.

    Parameters
    ----------
    i: int
        Integer to be encoded

    Returns
    -------
    np.array
        4d array holding the fizz buzz representation of the integer i
    """
    if i % 15 == 0: # pylint: disable=R1705
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


def encode_input_and_output(num_digits, n_max):
    """encode inputs and outputs
    """
    train_x = np.array([binary_encode(i, num_digits) for i in range(101, n_max + 1)])
    train_y = np.array([fizz_buzz_encode(i) for i in range(101, n_max + 1)])

    return train_x, train_y

def init_weights(shape, std_dev=0.03):
    """Function to initialize weights

    Parameters
    ----------
    shape: [int]
        List of dimensions of the layer to be initialized.
    std_dev: float
        Standard deviation of the distribution that's used to initialize the weights.

    Returns
    -------
    tf.Variable
        TF variable with initialized weights
    """
    return tf.Variable(tf.random_normal(shape, stddev=std_dev))

def create_model(inputs, num_digits, num_hidden):
    """Function to create TF model

    Parameters
    ----------
    num_digits: int
        Number of nodes in the input layer.
    num_hidden: int
        Number of nodes in the hidden layer
    """

    weights_hidden = init_weights([num_digits, num_hidden])
    weights_outpout = init_weights([num_hidden, 4])

    hidden_layer = tf.nn.relu(tf.matmul(inputs, weights_hidden))
    output_layer = tf.matmul(hidden_layer, weights_outpout)

    return output_layer

def decode_fizz_buzz(i, prediction):
    """Function to decode model prediction

    Parameters
    ----------
    i: int
        Input number
    prediciction: int
        Label predicted by the neural network

    Returns
    -------
    str
        Decoded ouput of either the number, fizz, buzz or fizzbuzz
    """
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def main(): # pylint: disable=R0914
    """main function
    """
    # create train sets
    train_x, train_y = encode_input_and_output(NUM_DIGITS, N_MAX)
    # create TF plaeholders
    X = tf.placeholder("float", [None, NUM_DIGITS]) # pylint: disable=C0103
    Y = tf.placeholder("float", [None, 4]) # pylint: disable=C0103
    # create TF model
    logits = create_model(X, NUM_DIGITS, NUM_HIDDEN)

    # pick cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
    y_pred = tf.nn.softmax(logits)
    predict_op = tf.argmax(y_pred, 1)

    actual = generate_fizz_buzz_sequence(100)
    actual_2 = generate_fizz_buzz_sequence(5000)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_loss = 0
            # randomize order of numbers
            perm = np.random.permutation(range(len(train_x)))
            train_x, train_y = train_x[perm], train_y[perm]

            for start in range(0, len(train_x), BATCH_SIZE):
                end = start + BATCH_SIZE
                _, batch_cost = sess.run([train_op, cost],
                                         feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                epoch_loss += batch_cost
            print("Epoch: {0}, train accuracy: {1}, epoch loss: {2}".format(
                epoch, np.mean(np.argmax(train_y, axis=1) ==
                               sess.run(predict_op, feed_dict={X: train_x, Y: train_y})),
                epoch_loss))

        numbers = np.arange(1, 101)
        test_x = np.transpose(binary_encode(numbers, NUM_DIGITS))

        test_y = sess.run(predict_op, feed_dict={X: test_x})
        output = np.vectorize(decode_fizz_buzz)(numbers, test_y)

        ### UNCOMMENT for additional model test

        # numbers_2 = np.arange(4097, 5000)
        # test_x_2 = np.transpose(binary_encode(numbers_2, NUM_DIGITS))
        #
        # test_y_2 = sess.run(predict_op, feed_dict={X: test_x_2})
        # output_2 = np.vectorize(decode_fizz_buzz)(numbers_2, test_y_2)

        print(output)
    correct = [(x == y) for x, y in zip(actual, output)]
    incorrect = [(x, y) for x, y in zip(actual, output) if x != y]
    print("Number of correct predictions:", sum(correct))
    print("Incorrect predictions: ", incorrect)

    ### UNCOMMENT for additional model test

    # correct_2 = [(x == y) for x, y in zip(actual_2[4096:], output_2)]
    # incorrect_2 = [(x, y) for x, y in zip(actual_2[4096:], output_2) if x != y]
    # print("Number of correct predictions:", sum(correct_2))
    # print("Incorrect predictions: ", incorrect_2)


if __name__ == "__main__":
    main()
