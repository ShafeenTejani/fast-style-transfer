import os

import numpy as np
import scipy
import scipy.misc
from os.path import exists
from sys import stdout

import math
from fast_style_transfer import FastStyleTransfer
from argparse import ArgumentParser
import tensorflow as tf
import transform

NETWORK_PATH='saved_networks'

def ffwd(content, network_path):
    with tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=content.shape,
                                         name='img_placeholder')

        network = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(network_path):
            ckpt = tf.train.get_checkpoint_state(network_path)
            print ckpt.model_checkpoint_path
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, network_path)

        prediction = sess.run(network, feed_dict={img_placeholder:content})
        return prediction[0]


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', type=str,
                        dest='content', help='content image path',
                        metavar='CONTENT', required=True)

    parser.add_argument('--network-path', type=str,
                        dest='network_path',
                        help='path to network (default %(default)s)',
                        metavar='NETWORK_PATH', default=NETWORK_PATH)

    parser.add_argument('--output-path', type=str,
                        dest='output_path',
                        help='path for output',
                        metavar='OUTPUT_PATH', required=True)

    return parser

def check_opts(opts):
    assert exists(opts.content), "content not found!"
    assert exists(opts.network_path), "network not found!"


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    network = options.network_path
    if not os.path.isdir(network):
        parser.error("Network %s does not exist." % network)

    content_image = load_image(options.content)
    content_image = np.ndarray.reshape(content_image, (1,) + content_image.shape)

    prediction = ffwd(content_image, network)
    save_image(options.output_path, prediction)

def load_image(image_path, img_size=None):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = scipy.misc.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_size is not None):
        img = imresize(img, img_size)

    img = img.astype("float32")
    return img


def save_image(path, img):
    print img.shape
    scipy.misc.imsave(path, np.clip(img, 0, 255).astype(np.uint8))

def print_losses(losses):
    stdout.write('  content loss: %g\n' % losses['content'])
    stdout.write('    style loss: %g\n' % losses['style'])
    stdout.write('       tv loss: %g\n' % losses['total_variation'])
    stdout.write('    total loss: %g\n' % losses['total'])


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

if __name__ == '__main__':
    main()
