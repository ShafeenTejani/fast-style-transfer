# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.

import os

import numpy as np
import scipy
import scipy.misc
from os.path import exists
from sys import stdout

import math
import utils
from fast_style_transfer import FastStyleTransfer
from argparse import ArgumentParser
import tensorflow as tf

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 5
TV_WEIGHT = 1e-6
LEARNING_RATE = 1e-3
NUM_EPOCHS=5
BATCH_SIZE=4
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CHECKPOINT_ITERATIONS = 100
SAVE_PATH = 'network'

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', required=True)

    parser.add_argument('--save-path', type=str,
                        dest='save_path',
                        help='directory to save network (default %(default)s)',
                        metavar='SAVE_PATH', default=SAVE_PATH)

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

    parser.add_argument('--use-gpu', dest='use_gpu', help='run on a GPU', action='store_true')
    parser.set_defaults(use_gpu=False)

    return parser

def check_opts(opts):
    assert exists(opts.style), "style path not found!"
    assert exists(opts.train_path), "train path not found!"

    assert exists(opts.vgg_path), "vgg network not found!"
    assert exists(opts.save_path), "save path not found!"
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_image = utils.load_image(options.style)
    style_image = np.ndarray.reshape(style_image, (1,) + style_image.shape)

    content_targets = utils.get_files(options.train_path)
    content_shape = utils.load_image(content_targets[0]).shape

    device = '/gpu:0' if options.use_gpu else '/cpu:0'

    style_transfer = FastStyleTransfer(
        vgg_path=VGG_PATH,
        style_image=style_image,
        content_shape=content_shape,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        tv_weight=options.style_weight,
        batch_size=options.batch_size,
        device=device)

    for iteration, network, first_image, losses in style_transfer.train(
        content_training_images=content_targets,
        learning_rate=options.learning_rate,
        epochs=options.epochs,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        print_losses(losses)

        saver = tf.train.Saver()
        if (iteration % 100 == 0):
            saver.save(network, opts.save_path + '/fast_style_network.ckpt')

        saver.save(network, opts.save_path + '/fast_style_network.ckpt')


def print_losses(losses):
    stdout.write('  content loss: %g\n' % losses['content'])
    stdout.write('    style loss: %g\n' % losses['style'])
    stdout.write('       tv loss: %g\n' % losses['total_variation'])
    stdout.write('    total loss: %g\n' % losses['total'])


if __name__ == '__main__':
    main()
