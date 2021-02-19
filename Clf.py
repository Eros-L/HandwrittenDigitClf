import sys, os
import struct
import numpy as np
import cv2 as cv
import tensorflow as tf


def load_mnist(path, kind='train'):
    """ load MNIST data from 'path' """
    label_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    image_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:
        magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def binary(x):
    images = np.reshape(x, (-1, 28, 28))
    for i in range(images.shape[0]):
        images[i] = cv.threshold(images[i], 63, 255, cv.THRESH_BINARY)[1]
    return np.reshape(images, (-1, 784))


def train_nn():
    x_train, y_train = load_mnist('Mnist')
    x_test, y_test = load_mnist('Mnist', 't10k')
    x_train = binary(x_train)
    x_test = binary(x_test)
    """ parameters """
    alpha = 1.0
    epochs = 25
    batch_size = 100
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(20).batch(batch_size).repeat()
    """ weight, bias """
    w1 = tf.Variable(tf.random_normal([784, 392], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([392]), name='b1')
    w2 = tf.Variable(tf.random_normal([392, 10], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')
    """ hidden layer """
    hidden_out = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
    """ output """
    out = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))
    """ loss """
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(tf.clip_by_value(out, 1e-10, 1.0)) +
                                                  (1-y) * tf.log(tf.clip_by_value(1-out, 1e-10, 1.0)), axis=1))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=alpha).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    """ initialization """
    init = tf.global_variables_initializer()
    iterator = dataset.make_initializable_iterator()
    data_element = iterator.get_next()
    """ one hot """
    y_train_one_hot = tf.one_hot(y_train, 10, 1, 0)
    y_test_one_hot = tf.one_hot(y_test, 10, 1, 0)
    """ saver """
    saver = tf.train.Saver(max_to_keep=1)
    """ session """
    with tf.Session() as sess:
        sess.run(init)
        y_train = sess.run(y_train_one_hot)
        y_test = sess.run(y_test_one_hot)
        sess.run(iterator.initializer, feed_dict={x: x_train, y: y_train})
        total_batch = len(y_train) // batch_size
        for epoch in range(epochs):
            for step in range(total_batch):
                x_batch, y_batch = sess.run(data_element)
                res, c = sess.run([optimizer, cross_entropy], feed_dict={x: x_batch, y: y_batch})
            print("Epoch:", (epoch + 1), "accuracy =", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
        saver.save(sess, 'Model/mnist')


def train_cnn():
    x_train, y_train = load_mnist('Mnist')
    x_test, y_test = load_mnist('Mnist', 't10k')
    x_train = binary(x_train)
    x_test = binary(x_test)
    """ parameters """
    alpha = 1.0
    epochs = 25
    batch_size = 100
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(20).batch(batch_size).repeat()
    """ weight, bias """
    w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1), name='w_conv1')
    b_conv1 = tf.Variable(tf.random_normal([6]), name='b_conv1')
    w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1), name='w_conv2')
    b_conv2 = tf.Variable(tf.random_normal([16]), name='b_conv2')
    w_fc1 = tf.Variable(tf.truncated_normal([7*7*16, 120], stddev=0.1), name='w_fc1')
    b_fc1 = tf.Variable(tf.random_normal([120]), name='b_fc1')
    w_fc2 = tf.Variable(tf.truncated_normal([120, 10], stddev=0.1), name='w_fc2')
    b_fc2 = tf.Variable(tf.random_normal([10]), name='b_fc2')
    """ hidden layer """
    h_conv1 = tf.nn.relu(tf.nn.conv2d(tf.reshape(x, [-1, 28, 28, 1]), w_conv1, strides=[1, 1, 1, 1], padding='SAME')
                         + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7*7*16]), w_fc1) + b_fc1)
    """ output """
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_conv = tf.nn.softmax(tf.matmul(tf.nn.dropout(h_fc1, keep_prob), w_fc2) + b_fc2, name='y_conv')
    """ loss """
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)) +
                                                  (1-y) * tf.log(tf.clip_by_value(1-y_conv, 1e-10, 1.0)), axis=1))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=alpha).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    """ initialization """
    init = tf.global_variables_initializer()
    iterator = dataset.make_initializable_iterator()
    data_element = iterator.get_next()
    """ one hot """
    y_train_one_hot = tf.one_hot(y_train, 10, 1, 0)
    y_test_one_hot = tf.one_hot(y_test, 10, 1, 0)
    """ saver """
    saver = tf.train.Saver(max_to_keep=1)
    """ session """
    with tf.Session() as sess:
        sess.run(init)
        y_train = sess.run(y_train_one_hot)
        y_test = sess.run(y_test_one_hot)
        sess.run(iterator.initializer, feed_dict={x: x_train, y: y_train})
        total_batch = len(y_train) // batch_size
        for epoch in range(epochs):
            for step in range(total_batch):
                x_batch, y_batch = sess.run(data_element)
                res, c = sess.run([optimizer, cross_entropy], feed_dict={x: x_batch, y: y_batch, keep_prob: 0.5})
            print("Epoch:", (epoch + 1), "accuracy =",
                  sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0}))
        saver.save(sess, 'Model/mnist')


def clf():
    if not os.path.isfile('Model/checkpoint'):
        train_cnn()
    graph = tf.get_default_graph()
    """ session """
    with tf.Session(graph=graph) as sess:
        """ loader """
        loader = tf.train.import_meta_graph('Model/mnist.meta')
        loader.restore(sess, tf.train.latest_checkpoint('Model'))
        """ tensor """
        x = graph.get_tensor_by_name('x:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        out = graph.get_tensor_by_name('y_conv:0')
        """ get the output of different cases """
        res = open('Result/stage2.csv', mode='r')
        lines = res.readlines()
        res.close()
        res = open('Result/stage2.csv', mode='w')
        res.write(lines[0])
        lines.pop(0)
        for i in range(0, 63):
            res.write(lines[0][:-1])
            lines.pop(0)
            row = 0
            while os.path.isdir('Segmentation/%d/%d' % (i, row)):
                col = 0
                digits = ''
                while os.path.isfile('Segmentation/%d/%d/%d.jpg' % (i, row, col)):
                    img = cv.imread('Segmentation/%d/%d/%d.jpg' % (i, row, col))
                    grey = 255 - cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    if grey.shape[0] > grey.shape[1]:
                        [height, width] = [20, int(20*grey.shape[1]/grey.shape[0])]
                        scale = cv.copyMakeBorder(cv.resize(grey, (width, height)), 4, 4, 14-int(width/2),
                                                  14-width+int(width/2), cv.BORDER_CONSTANT, value=0)
                    else:
                        [height, width] = [int(20*grey.shape[0]/grey.shape[1]), 20]
                        scale = cv.copyMakeBorder(cv.resize(grey, (width, height)), 14-int(height/2),
                                                  14-height+int(height/2), 4, 4, cv.BORDER_CONSTANT, value=0)
                    scale = cv.threshold(scale, 63, 255, cv.THRESH_BINARY)[1]
                    # cv.imwrite('%d.jpg' % col, scale)
                    digits += str(sess.run(tf.argmax(out, 1), feed_dict={x: np.reshape(scale, (1, 784)), keep_prob: 1.0})[0])
                    col += 1
                res.write(digits + ',')
                row += 1
            res.write('\n')
        res.close()


def main():
    clf()


if __name__ == '__main__':
    main()
