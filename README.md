Work in progress

# About

This script generates new images of cats using the technique of generative adversarial networks (GAN), as described in the paper by [Ian J. Goodfellow](http://arxiv.org/abs/1406.2661).
It is based on code from facebook's [eyescream project](https://github.com/facebook/eyescream).

The basic principle of GANs is to train two networks in a kind of pupil teacher relationship.
The pupil is called G (generator) and the teacher D (discriminator).
G generates new images, D tells fake images from real images apart.
They are both jointly trained, i.e. G's training objective is to trick D into believing that its outputs are real images, while D is trained to spot G's fakes.
At the end you (hopefully) get a G that produces beautiful images matching your training dataset.

# Images

- gif progress per epoch

- color: end result, random images + training set comparison

- grayscale: end result, random images + training set comparison

# Requirements

* [Torch](http://torch.ch/) with the following libraries (most of them are probably already installed by default):
  * `nn` (`luarocks install nn`)
  * `paths` (`luarocks install paths`)
  * `image` (`luarocks install image`)
  * `optim` (`luarocks install optim`)
  * `cutorch` (`luarocks install cutorch`)
  * `cunn` (`luarocks install cunn`)
  * `dpnn` (`luarocks install dpnn`)
* [display](https://github.com/szym/display)
* [10k cats dataset](https://web.archive.org/web/20150520175645/http://137.189.35.203/WebUI/CatDatabase/catData.html)
* GPU strongly recommended (nothing below 3GB memory, only Nvidia)

- todo all requirements?

# Usage

- install requirements
- download and extract dataset to some filepath /foo/bar so that /foo/bar contains directories CATS_00 to CAT_06 (note: /foo/bar should be on an SSD)
- clone rep
- copy segmentation script to /foo/bar
- run segmentation script, creates new folder /foo/bar/out_faces_64x64
- set your dataset path in all files to /foo/bar/out_faces_64x64
- start display
- train v for 100 epochs or so (~20min)
- pretrain g for 100 epochs or so (~20min)
- train
- upscale

# V

# Architecture

- current architecture D
- current architecture G
- current architecture V

- something about d iterations

- adam
