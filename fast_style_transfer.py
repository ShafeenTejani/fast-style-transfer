import vgg_network

import tensorflow as tf
import numpy as np
import scipy

from sys import stdout
from functools import reduce
from os.path import exists
import transform


class FastStyleTransfer:
    CONTENT_LAYER = 'relu4_2'
    STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

    def __init__(self, vgg_path,
                style_image, content_weight,
                style_weight, tv_weight, batch_size):
        vgg = vgg_network.VGG(vgg_path)
        self.style_image = style_image
        self.batch_size = batch_size
        self.batch_shape = (batch_size,80,160,3)

        self.input_batch = tf.placeholder(tf.float32, shape=self.batch_shape, name="input_batch")

        content_net = vgg.net(vgg.preprocess(self.input_batch))

        style_image_placeholder = tf.placeholder('float', shape=style_image.shape)
        style_net = vgg.net(style_image_placeholder)

        self.stylized_image = transform.net(self.input_batch/255.0)

        stylized_net = vgg.net(vgg.preprocess(self.stylized_image))

        self.content_loss = content_weight * (2 * tf.nn.l2_loss(
                content_net[self.CONTENT_LAYER] - stylized_net[self.CONTENT_LAYER]) /
                (_tensor_size(content_net[self.CONTENT_LAYER]) * batch_size))

        with tf.Session() as sess:
            style_loss = 0
            style_preprocessed = vgg.preprocess(style_image)

            for layer in self.STYLE_LAYERS:
                style_image_gram = calculate_style_gram_matrix_for(style_net,
                                                                   style_image_placeholder,
                                                                   layer,
                                                                   style_preprocessed)

                input_image_gram = calculate_input_gram_matrix_for(stylized_net, layer)

                style_loss += (2 * tf.nn.l2_loss(input_image_gram - style_image_gram) / style_image_gram.size)

            self.style_loss = style_weight * (style_loss / batch_size)


        self.total_variation_loss = tv_loss(self.stylized_image, self.batch_shape, tv_weight) / batch_size

        self.loss = self.content_loss  + self.style_loss + self.total_variation_loss


    def _current_loss(self, feed_dict):
        losses = {}
        losses['content'] = self.content_loss.eval(feed_dict=feed_dict)
        losses['style'] = self.style_loss.eval(feed_dict=feed_dict)
        losses['total_variation'] = self.total_variation_loss.eval(feed_dict=feed_dict)
        losses['total'] = self.loss.eval(feed_dict=feed_dict)
        return losses

    def train(self, content_training_images,
        learning_rate, epochs, checkpoint_iterations):

        def is_checkpoint_iteration(i):
            return (checkpoint_iterations and i % checkpoint_iterations == 0)

        def print_progress(i):
            stdout.write('Iteration %d\n' % (i + 1))

        best_loss = float('inf')
        best = None

        with tf.Session() as sess:
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            sess.run(tf.initialize_all_variables())
            iterations = 0
            for epoch in range(epochs):
                for i in range(0, len(content_training_images), self.batch_size):
                    print_progress(iterations)

                    batch = np.zeros(self.batch_shape, dtype=np.float32)

                    for j, img_path in enumerate(content_training_images[i: i+self.batch_size]):
                        batch[j] = load_image(img_path, img_size=self.batch_shape[1:])

                    train_step.run(feed_dict={self.input_batch:batch})

                    if is_checkpoint_iteration(iterations):
                        yield (
                            iterations,
                            sess,
                            self.stylized_image.eval(feed_dict={self.input_batch:batch})[0],
                            self._current_loss({self.input_batch:batch})
                       )
                    iterations += 1

def calculate_style_gram_matrix_for(network, image, layer, style_image):
    image_feature = network[layer].eval(feed_dict={image: style_image})
    image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
    return np.matmul(image_feature.T, image_feature) / image_feature.size

def calculate_input_gram_matrix_for(network, layer):
    image_feature = network[layer]
    batch_size, height, width, number = map(lambda i: i.value, image_feature.get_shape())
    size = height * width * number
    image_feature = tf.reshape(image_feature, (batch_size, height * width, number))
    return tf.batch_matmul(tf.transpose(image_feature, perm=[0,2,1]), image_feature) / size

def tv_loss(image, shape, tv_weight):
    # total variation denoising
    tv_y_size = _tensor_size(image[:,1:,:,:])
    tv_x_size = _tensor_size(image[:,:,1:,:])
    tv_loss = tv_weight * 2 * (
            (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                tv_y_size) +
            (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                tv_x_size))

    return tv_loss

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def load_image(image_path, img_size=None):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = scipy.misc.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_size is not None):
        img = scipy.misc.imresize(img, img_size)

    img = img.astype("float32")
    return img

# wrapper
# transform network
