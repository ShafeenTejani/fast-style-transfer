# Real-Time Style Transfer
A TensorFlow implementation of real-time style transfer based on the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Johnson et. al

## Algorithm

See my related blog post(link) for an overview of the algorithm for real-time style transfer.

The total loss used is the weighted sum of the style loss, the content loss and a total variation loss. This third component is not specfically mentioned in the original paper but leads to more cohesive images being generated.

## Requirements

* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup#download-and-setup)
* [SciPy & NumPy](http://scipy.org/install.html)
* Download the [pre-trained VGG network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) and place it in the top level of the repository (~500MB)
* For training: 
  * It is recommended to use a GPU to get good results within a reasonable timeframe
  * You will need an image dataset to train your networks. I used the [Microsoft COCO dataset](http://mscoco.org/) and resized the images to 256x256 pixels
* Generation of styled images can be run on a CPU or GPU. Some pre-trained style networks can be download from [here](https://drive.google.com/open?id=0B7pvkmVwDrF8a3FCVUt5RGhQSlU) (~700MB)

## Running the code

### Training a network for a particuar style

```python run.py --content <content image> --style <style image> --output <output image path>```

The algorithm will run with the following settings:

```python 
ITERATIONS = 1000    # override with --iterations argument
LEARNING_RATE = 1e1  # override with --learning-rate argument
CONTENT_WEIGHT = 5e1 # override with --content-weight argument
STYLE_WEIGHT = 1e2   # override with --style-weight argument
TV_WEIGHT = 1e2      # override with --tv-weight argument
```

By default the style transfer will start with a random noise image and optimise it to generate an output image. To start with a particular image (for example the content image) run with the `--initial <initial image>` argument.
    
To run the style transfer with a GPU run with the `--use-gpu` flag.

### Using a trained network to generate a style transfer

```do some stuff here```

I have made 3 3 pre-trained networks for the 3 styles shown in the results section below available. They can be downloaded from [here](https://drive.google.com/open?id=0B7pvkmVwDrF8a3FCVUt5RGhQSlU) (~700MB).

## Results

I trained three networks style transfers using the following three style images:

![Style Images](results/style_images.png)

Each network was trained with 80,000 training images taken from the Microsoft COCO dataset and resized to 256×256 pixels. Training was carried out for 100,000 iterations with a batch size of 4 and took approximately 12 hours on a GTX 1080 GPU. Here are some of the style transfers I was able to generate:

![Results](results/style_transfers.png)

## Acknowledgements

This code was inspired by an existing TensorFlow [implementation by Logan Engstrom](https://github.com/lengstrom/fast-style-transfer), and I have re-used most of his transform network code here.
The VGG network code is based on an existing [implementation by Anish Anish Athalye](https://github.com/anishathalye/neural-style)

## License

Released under GPLv3, see [LICENSE.txt](LICENSE.txt)
