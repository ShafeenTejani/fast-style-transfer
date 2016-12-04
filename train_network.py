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

# default arguments
CONTENT_WEIGHT = 7.5
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3
NUM_EPOCHS=2000
BATCH_SIZE=3
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CHECKPOINT_ITERATIONS = 1
TRAIN_PATH = 'train_simple'

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

def check_opts(opts):
    assert exists(opts.style), "style path not found!"
    assert exists(opts.train_path), "train path not found!"

    assert exists(opts.vgg_path), "vgg network data not found!"
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

def _get_files(img_dir):
    files = list_files(img_dir)
    return map(lambda x: os.path.join(img_dir,x), files)

def load_image(image_path, img_size=None):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = scipy.misc.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_size is not None):
        img = imresize(img, img_size)

    img = img.astype("float32")
    return img


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    if not os.path.isfile(VGG_PATH):
        parser.error("Network %s does not exist." % VGG_PATH)

    style_image = load_image(options.style)
    style_image = np.ndarray.reshape(style_image, (1,) + style_image.shape)

    content_targets = _get_files(options.train_path)


    style_transfer = FastStyleTransfer(
        vgg_path=VGG_PATH,
        style_image=style_image,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        tv_weight=options.style_weight,
        batch_size=options.batch_size)

    for iteration, network, first_image, losses in style_transfer.train(
        content_training_images=content_targets,
        learning_rate=options.learning_rate,
        epochs=options.epochs,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        print_losses(losses)

        saver = tf.train.Saver()
        if (iteration % 100 == 0):
            saver.save(network, 'networks/fast_style_network.ckpt')
        save_image(first_image, 'outputs/iteration_' + str(iteration) + '.png')


def save_image(img, path):
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
