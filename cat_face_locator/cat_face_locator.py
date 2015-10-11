# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import re
import os
import gzip
import sys
import random
import numpy as np
import pandas
import csv
import math
from scipy import misc
from skimage import color
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, TimeDistributedDense, RepeatVector, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils.generic_utils import Progbar
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.recurrent import GRU, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from ImageAugmenter import ImageAugmenter
import argparse
from saveload import load_weights_seq

os.sys.setrecursionlimit(10000)
np.random.seed(42)
random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MAIN_DIR = "/media/aj/grab/ml/datasets/10k_cats"
DIRS = [os.path.join(MAIN_DIR, subdir) for subdir in ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]]
VISUALIZE = False

MODEL_IMAGE_HEIGHT = 64
MODEL_IMAGE_WIDTH = 64
GRAYSCALE = False
SAVE_EXAMPLES = True
SAVE_EXAMPLES_DIR = os.path.join(CURRENT_DIR, "examples")
SAVE_WEIGHTS_FILEPATH = os.path.join(CURRENT_DIR, "cat-face-detector_tiny.weights")
SAVE_AUTO_OVERWRITE = True

def main():
    parser = argparse.ArgumentParser(description="Generate data for a word")
    parser.add_argument("--load", required=False, help="Reload weights from file")
    args = parser.parse_args()
    
    # numbers here will be increased by augmentations,
    # i.e. at 5 augmentations per image 9400 + 5*9400 (original + augmented)
    n_train = 9400
    n_val = 512
    """
    X, Y = get_examples(n_train+n_val, augmentations=5)
    X_train = X[0:n_train, ...]
    Y_train = Y[0:n_train, ...]
    X_val = X[n_train:, ...]
    Y_val = Y[n_train:, ...]
    """
    X_train, Y_train = get_examples(n_train, augmentations=5)
    X_val, Y_val = get_examples(n_val, start_at=n_train, augmentations=1)
    print("Collected examples:")
    print("X_train: ", X_train.shape)
    print("X_val:", X_val.shape)
    print("Y_train:", Y_train.shape)
    print("Y_val:", Y_val.shape)
    
    
    # debug: show training images with marked faces
    """
    for i, image in enumerate(X_train):
        print("-----------_")
        print(image.shape)
        print(Y_train[i])
        tl_y, tl_x, br_y, br_x = center_scale_to_pixels(image, Y_train[i][0], Y_train[i][1], Y_train[i][2], Y_train[i][3])
        marked_image = visualize_rectangle(image*255, tl_x, br_x, tl_y, br_y, (255,), channel_is_first_axis=True)
        misc.imshow(np.squeeze(marked_image))
    """
    
    model = create_model_tiny(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, Adam())
    
    if args.load:
        print("Loading weights...")
        load_weights_seq(model, args.load)

    model.fit(X_train, Y_train, batch_size=128, nb_epoch=50, validation_split=0.0, validation_data=(X_val, Y_val), show_accuracy=False)

    print("Saving weights...")
    model.save_weights(SAVE_WEIGHTS_FILEPATH, overwrite=SAVE_AUTO_OVERWRITE)

    if SAVE_EXAMPLES:
        print("Saving examples (predictions)...")
        y_preds = predict_on_images(model, X_val)
        for img_idx, (tl_y, tl_x, br_y, br_x) in enumerate(y_preds):
            image = np.rollaxis(X_val[img_idx, ...], 0, 3)
            if GRAYSCALE:
                image_marked = visualize_rectangle(image*255, tl_x, br_x, tl_y, br_y, (255,), channel_is_first_axis=False)
            else:
                image_marked = visualize_rectangle(image*255, tl_x, br_x, tl_y, br_y, (255,0,0), channel_is_first_axis=False)
            #misc.imshow(np.squeeze(image_marked))
            misc.imsave(os.path.join(SAVE_EXAMPLES_DIR, "%d.png" % (img_idx,)), np.squeeze(image_marked))
            
    """
    y_preds = predict_on_sentences(model, sentences)
    sentences_pred = CHAR_VECTORIZER.reverse_transform_maxval(y_preds)
    for sentence, y_pred, sentence_pred in zip(sentences, y_preds, sentences_pred):
        print("*", sentence)
        print(" ==> ", sentence_pred)
        print(" ==> ", y_pred)
    """

def create_model_tiny(image_height, image_width, optimizer):
    model = Sequential()
    
     # 3x64x64
    model.add(Convolution2D(4, 1 if GRAYSCALE else 3, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(4, 4, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # 4x15x15
    new_image_height = (((image_height - 2) / 2) - 2) / 2
    new_image_height = int(new_image_height)
    
    new_image_width = (((image_width - 2) / 2) - 2) / 2
    new_image_width = int(new_image_width)
    
    nb_last_kernels = 4
    flat_size = nb_last_kernels * new_image_height * new_image_width
    
    model.add(Flatten())
    
    #model.add(BatchNormalization((flat_size)))
    model.add(Dense(flat_size, 64))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, 64))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, 4))
    model.add(Activation("sigmoid"))
    
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def create_model_small(image_height, image_width, optimizer):
    model = Sequential()
    
     # 3x64x64
    model.add(Convolution2D(32, 1 if GRAYSCALE else 3, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(32, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 64x30x30
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 64x13x13
    new_image_height = (image_height - 2 - 2) / 2
    new_image_height = (new_image_height - 2 - 2) / 2
    new_image_height = int(new_image_height)
    
    new_image_width = (image_width - 2 - 2) / 2
    new_image_width = (new_image_width - 2 - 2) / 2
    new_image_width = int(new_image_width)
    
    nb_last_kernels = 64
    
    model.add(Reshape(nb_last_kernels, new_image_height * new_image_width))
    
    """
    model.add(BatchNormalization(flattened_size))
    model.add(Dense(flattened_size, 256, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(256))
    model.add(Dropout(0.5))
    """
    model.add(BatchNormalization((nb_last_kernels, new_image_height * new_image_width)))
    model.add(LSTM(new_image_height * new_image_width, 128, return_sequences=True))
    #model.add(Flatten())
    model.add(Dropout(0.5))
    
    """
    model.add(BatchNormalization(64*32))
    #model.add(Dense(256, 256, W_regularizer=l2(0.0001)))
    model.add(Dense(64*32, 1024, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(1024))
    model.add(Dropout(0.5))
    """
    
    model.add(BatchNormalization((nb_last_kernels, 128)))
    model.add(LSTM(128, 32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(nb_last_kernels * 32, 4))
    model.add(Activation("sigmoid"))
    
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def create_model(image_height, image_width, optimizer):
    model = Sequential()
    
     # 3x64x64
    model.add(Convolution2D(16, 1 if GRAYSCALE else 3, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(16, 16, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 16x30x30
    model.add(Convolution2D(32, 16, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(32, 32, 3, 3, border_mode="same"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 32x16x16
    model.add(Convolution2D(128, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 128, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 128x6x6
    new_image_height = (image_height - 2 - 2) / 2
    new_image_height = (new_image_height + 2 + 0) / 2
    new_image_height = (new_image_height - 2 - 2) / 2
    new_image_height = int(new_image_height)
    
    new_image_width = (image_width - 2 - 2) / 2
    new_image_width = (new_image_width + 2 + 0) / 2
    new_image_width = (new_image_width - 2 - 2) / 2
    new_image_width = int(new_image_width)
    
    nb_last_kernels = 128
    
    model.add(Reshape(nb_last_kernels, new_image_height * new_image_width))
    
    """
    model.add(BatchNormalization(flattened_size))
    model.add(Dense(flattened_size, 256, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(256))
    model.add(Dropout(0.5))
    """
    model.add(BatchNormalization((nb_last_kernels, new_image_height * new_image_width)))
    model.add(LSTM(new_image_height * new_image_width, 256, return_sequences=True))
    #model.add(Flatten())
    model.add(Dropout(0.5))
    
    """
    model.add(BatchNormalization(64*32))
    #model.add(Dense(256, 256, W_regularizer=l2(0.0001)))
    model.add(Dense(64*32, 1024, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(1024))
    model.add(Dropout(0.5))
    """
    
    model.add(BatchNormalization((nb_last_kernels, 256)))
    model.add(LSTM(256, 32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(nb_last_kernels * 32, 4))
    model.add(Activation("sigmoid"))
    
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def create_model2(image_height, image_width, optimizer):
    model = Sequential()
    
     # 3x64x64
    model.add(Convolution2D(16, 1 if GRAYSCALE else 3, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(16, 16, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 16x30x30
    model.add(Convolution2D(32, 16, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(32, 32, 3, 3, border_mode="same"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 32x16x16
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 64x12x12
    model.add(Convolution2D(128, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 128, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 128x8x8
    new_image_height = (image_height - 2 - 2) / 2
    new_image_height = (new_image_height + 2 + 0) / 2
    new_image_height = (new_image_height - 2 - 2)
    new_image_height = (new_image_height - 2 - 2)
    new_image_height = int(new_image_height)
    
    new_image_width = (image_width - 2 - 2) / 2
    new_image_width = (new_image_width + 2 + 0) / 2
    new_image_width = (new_image_width - 2 - 2)
    new_image_width = (new_image_width - 2 - 2)
    new_image_width = int(new_image_width)
    
    nb_last_kernels = 128
    
    model.add(Reshape(nb_last_kernels, new_image_height * new_image_width))
    
    """
    model.add(BatchNormalization(flattened_size))
    model.add(Dense(flattened_size, 256, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(256))
    model.add(Dropout(0.5))
    """
    model.add(BatchNormalization((nb_last_kernels, new_image_height * new_image_width)))
    model.add(LSTM(new_image_height * new_image_width, 128, return_sequences=True))
    #model.add(Flatten())
    model.add(Dropout(0.5))
    
    """
    model.add(BatchNormalization(64*32))
    #model.add(Dense(256, 256, W_regularizer=l2(0.0001)))
    model.add(Dense(64*32, 1024, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(1024))
    model.add(Dropout(0.5))
    """
    
    model.add(BatchNormalization((nb_last_kernels, 128)))
    model.add(LSTM(128, 32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(nb_last_kernels * 32, 4))
    model.add(Activation("sigmoid"))
    
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    #model.compile(loss="mae", optimizer=optimizer)
    return model

def create_model3(image_height, image_width, optimizer):
    alpha = 0.5
    model = Sequential()
    
     # 3x64x64
    model.add(Convolution2D(16, 1 if GRAYSCALE else 3, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(16, 16, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 16x30x30
    model.add(Convolution2D(32, 16, 3, 3, border_mode="full"))
    model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(32, 32, 3, 3, border_mode="same"))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 32x16x16
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2)))
    #model.add(GaussianNoise(0.05))
    model.add(Dropout(0.5))
    
    # 64x6x6
    new_image_height = (image_height - 2 - 2) / 2
    new_image_height = (new_image_height + 2 + 0) / 2
    new_image_height = (new_image_height - 2 - 2) / 2
    new_image_height = int(new_image_height)
    
    new_image_width = (image_width - 2 - 2) / 2
    new_image_width = (new_image_width + 2 + 0) / 2
    new_image_width = (new_image_width - 2 - 2) / 2
    new_image_width = int(new_image_width)
    
    nb_last_kernels = 64
    
    model.add(Reshape(nb_last_kernels, new_image_height * new_image_width))
    
    """
    model.add(BatchNormalization(flattened_size))
    model.add(Dense(flattened_size, 256, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(256))
    model.add(Dropout(0.5))
    """
    model.add(BatchNormalization((nb_last_kernels, new_image_height * new_image_width)))
    model.add(LSTM(new_image_height * new_image_width, 128, return_sequences=True))
    #model.add(Flatten())
    model.add(Dropout(0.5))
    
    """
    model.add(BatchNormalization(64*32))
    #model.add(Dense(256, 256, W_regularizer=l2(0.0001)))
    model.add(Dense(64*32, 1024, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization(1024))
    model.add(Dropout(0.5))
    """
    
    model.add(BatchNormalization((nb_last_kernels, 128)))
    model.add(LSTM(128, 32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(nb_last_kernels * 32, 4))
    model.add(Activation("sigmoid"))
    
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    #model.compile(loss="mae", optimizer=optimizer)
    return model

def get_all_filepaths(fp_dirs):
    result_img = []
    result_coords = []
    for fp_dir in fp_dirs:
        fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
        fps = [os.path.join(fp_dir, f) for f in fps]
        #print(len(fps))
        #print(fps)
        fps_img = [fp for fp in fps if re.match(r".*\.jpg$", fp)]
        fps_coords = [fp for fp in fps if re.match(r".*\.jpg\.cat$", fp)]
        result_img.extend(fps_img)
        #result_coords.extend(fps_coords)

    return result_img #, result_coords

def get_image_with_rectangle(image_filepath, coords_filepath):
    #fp_img_cat = "%s.cat" % (fp_img)
    fp_img = image_filepath
    fp_img_cat = coords_filepath
    if not os.path.isfile(fp_img) or not os.path.isfile(fp_img_cat):
        print("[WARNING] Either '%s' or '%s' could not be found" % (fp_img, fp_img_cat))
        return None, (None, None), (None, None)
    
    filename = fp_img[fp_img.rfind("/")+1:]
    image = misc.imread(fp_img, flatten=GRAYSCALE)
    """
    if GRAYSCALE:
        image_tmp = np.zeros((image.shape[0], image.shape[1], 1))
        image_tmp[:, :, 0] = image
        image = image_tmp
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    coords_raw = open(fp_img_cat, "r").readlines()[0].strip().split(" ")
    coords_raw = [abs(int(coord)) for coord in coords_raw]
    coords = []
    for i in range(0, len(coords_raw[1:]) - 1, 2): # first element is the number of coords
        x = min(image_width-1, max(0, coords_raw[i+1]))
        y = min(image_height-1, max(0, coords_raw[i+2]))
        pair = (x, y)
        coords.append(pair)
    
    # coords: 0 = left eye, 1 = right eye, 2 = nose
    face_center_x = (coords[0][0] + coords[1][0] + coords[2][0]) / 3
    face_center_y = (coords[0][1] + coords[1][1] + coords[2][1]) / 3
    face_center = (int(face_center_x), int(face_center_y))
    
    # mark all coordinates
    #print(coords_raw)
    if VISUALIZE:
        image_marked = np.copy(image)
        for coord in coords:
            #print("marking ", coord[0], coord[1])
            if GRAYSCALE:
                image_marked[coord[1], coord[0]] = 255
            else:
                image_marked[coord[1], coord[0], 0] = 255
    
    # 1st rectangle: convex hull (as rectangle) around provided coordinates
    min_x = min([coord[0] for coord in coords])
    min_y = min([coord[1] for coord in coords])
    max_x = max([coord[0] for coord in coords])
    max_y = max([coord[1] for coord in coords])
    rectangle_center = (min_x + (max_x - min_x)/2, min_y + (max_y - min_y)/2)
    rectangle_center = (int(rectangle_center[0]), int(rectangle_center[1]))
    center_diff = (face_center[0] - rectangle_center[0], face_center[1] - rectangle_center[1])
    
    # 2nd rectangle: the same rectangle as 1st, but translated to the center of the face
    min_x_fcenter = max(0, min_x + center_diff[0])
    min_y_fcenter = max(0, min_y + center_diff[1])
    max_x_fcenter = min(image_width-1, max_x + center_diff[0])
    max_y_fcenter = min(image_height-1, max_y + center_diff[1])
    
    # 3rd rectangle: the same rectangle as 1st, but translated _half-way_ towards the center of the face
    min_x_half = max(0, int(min_x + center_diff[0]/2))
    min_y_half = max(0, int(min_y + center_diff[1]/2))
    max_x_half = min(image_width-1, int(max_x + center_diff[0]/2))
    max_y_half = min(image_height-1, int(max_y + center_diff[1]/2))
    
    # 4th rectangle: a merge between 1st and 3rd rectangle, essentially a convex hull around
    # the corners of both rectangles
    min_x_merge = max(0, min(min_x, min_x_half))
    min_y_merge = max(0, min(min_y, min_y_half))
    max_x_merge = min(image_width-1, max(max_x, max_x_half))
    max_y_merge = min(image_height-1, max(max_y, max_y_half))
    
    # 5th rectangle: like 4th, but decreased in rows/columns to be squared
    rec4_height = max_y_merge - min_y_merge
    rec4_width = max_x_merge - min_x_merge
    min_x_merge_sq = min_x_merge
    max_x_merge_sq = max_x_merge
    min_y_merge_sq = min_y_merge
    max_y_merge_sq = max_y_merge
    if rec4_height == rec4_width:
        #print("A")
        pass
    elif rec4_height > rec4_width:
        #print("B")
        diff = rec4_height - rec4_width
        remove_top = math.floor(diff / 2)
        remove_bottom = math.floor(diff / 2)
        if diff % 2 != 0:
            remove_top += 1
        min_y_merge_sq += int(remove_top)
        max_y_merge_sq -= int(remove_bottom)
        #print("remove_top, remove_bottom", remove_top, remove_bottom)
    elif rec4_width > rec4_height:
        #print("C")
        diff = rec4_width - rec4_height
        remove_left = math.floor(diff / 2)
        remove_right = math.floor(diff / 2)
        if diff % 2 != 0:
            remove_left += 1
        min_x_merge_sq += int(remove_left)
        max_x_merge_sq -= int(remove_right)
        #print("remove_left, remove_right", remove_left, remove_right)
    #print("min_x_merge, max_x_merge, min_y_merge, max_y_merge", min_x_merge, max_x_merge, min_y_merge, max_y_merge)
    #print("min_x_merge_sq, max_x_merge_sq, min_y_merge_sq, max_y_merge_sq", min_x_merge_sq, max_x_merge_sq, min_y_merge_sq, max_y_merge_sq)
    
    # ------------------------
    
    if VISUALIZE:
        # visualize 1st rectangle in green
        image_marked = visualize_rectangle(image_marked, min_x, max_x, min_y, max_y, (0, 255, 0))
        
        # visualize 2nd rectangle in blue
        image_marked = visualize_rectangle(image_marked, min_x_fcenter, max_x_fcenter, min_y_fcenter, max_y_fcenter, (0, 0, 255))
        
        # visualize 3rd rectangle in red
        image_marked = visualize_rectangle(image_marked, min_x_half, max_x_half, min_y_half, max_y_half, (255, 0, 0))
        
        # visualize 4th rectangle in yellow
        image_marked = visualize_rectangle(image_marked, min_x_merge, max_x_merge, min_y_merge, max_y_merge, (255, 255, 0))
            
        # visualize 5th rectangle in cyan
        image_marked = visualize_rectangle(image_marked, min_x_merge_sq, max_x_merge_sq, min_y_merge_sq, max_y_merge_sq, (0, 255, 255))

        misc.imshow(image_marked)
    # -----------------------

    # pad the image around the borders so that it is square
    #print("Coords before squaring: y=", min_y_merge_sq, 
    image, (pad_top, pad_right, pad_bottom, pad_left) = square_image(image)
    image_height_square = image.shape[0]
    image_width_square = image.shape[1]
    min_x_merge_sq += pad_left
    max_x_merge_sq += pad_left
    min_y_merge_sq += pad_top
    max_y_merge_sq += pad_top
    
    #print("@get_image_with_rectangle after image padding")
    #print("Coords:", min_x_merge_sq, max_x_merge_sq, min_y_merge_sq, max_y_merge_sq)
    #marked_image = visualize_rectangle(image, min_x_merge_sq, max_x_merge_sq, min_y_merge_sq, max_y_merge_sq, (255,), channel_is_first_axis=False)
    #misc.imshow(np.squeeze(marked_image))
    # -------------

    # calculate center and scales of the face box
    rect_height = max_y_merge_sq - min_y_merge_sq
    rect_width = max_x_merge_sq - min_x_merge_sq
    rect_scale_y = rect_height / 2
    rect_scale_x = rect_width / 2
    rect_center_y = min_y_merge_sq + rect_scale_y
    rect_center_x = min_x_merge_sq + rect_scale_x

    #print(rect_center_y, "of", image.shape[0], rect_center_y/image.shape[0])
    #print(rect_center_x, "of", image.shape[1], rect_center_x/image.shape[1])
    #print(rect_scale_y, rect_scale_x)

    # resize image to 32x32
    image = misc.imresize(image, (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))

    if GRAYSCALE:
        image_tmp = np.zeros((image.shape[0], image.shape[1], 1))
        image_tmp[:, :, 0] = image
        image = image_tmp
    
    return image, (rect_center_y/image_height_square, rect_center_x/image_width_square), (rect_scale_y/image_height_square, rect_scale_x/image_width_square)

def square_image(image):
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    #print("Start at ", image_height, " / ", image_width)
    
    idx = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    
    # loops here are inefficient, but easy to read
    while image_width < image_height:
        if idx % 2 == 0:
            pad_left += 1
        else:
            pad_right += 1
        image_width += 1
        idx += 1
    
    idx = 0
    while image_height < image_width:
        if idx % 2 == 0:
            pad_top += 1
        else:
            pad_bottom += 1
        image_height += 1
        idx += 1
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if GRAYSCALE:
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=str("median"))
        else:
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=str("median"))

    #print("End at ", image_height, " / ", image_width)
    #print("Padding:", pad_top, pad_right, pad_bottom, pad_left)

    return image, (pad_top, pad_right, pad_bottom, pad_left)

def get_examples(count, start_at=0, augmentations=0):
    # low strength augmentation because we will not change the coordinates, so the image
    # should be kept mostly the same
    ia = ImageAugmenter(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH,
                        channel_is_first_axis=False,
                        hflip=False, vflip=False,
                        scale_to_percent=(0.95, 1.05), scale_axis_equally=True,
                        rotation_deg=5, shear_deg=2,
                        translation_x_px=1, translation_y_px=1)
    
    images_filepaths = get_all_filepaths(DIRS)
    images = []
    labels = []
    for image_filepath in images_filepaths[start_at:start_at+count]:
        coords_filepath = "%s.cat" % (image_filepath,)
        image, (center_y, center_x), (scale_y, scale_x) = get_image_with_rectangle(image_filepath, coords_filepath)
        # catch images with missing coordinates
        if image is not None:
            images.append(image / 255) # project pixel values to 0-1
            y = [center_y, center_x, scale_y, scale_x]
            labels.append(y)
            
            if augmentations > 0:
                images_aug = []
                for i in range(augmentations):
                    if i % 2 == 0:
                        images_aug.append(np.fliplr(image))
                        labels.append((center_y, 1-center_x, scale_y, scale_x))
                    else:
                        images_aug.append(image)
                        labels.append(y)
                images_aug = ia.augment_batch(np.array(images_aug, dtype=np.uint8)) # also projects pixel values to 0-1
                images.extend(images_aug)
    
    #print(images)
    #images = np.array(images, dtype=np.uint8)
    #images = ia.augment_batch(images) # project pixel values to 0-1
    images = np.array(images, dtype=np.float32)
    images = np.rollaxis(images, 3, 1)
    
    return images, np.array(labels, dtype=np.float32)

def center_scale_to_pixels(image, center_y, center_x, scale_y, scale_x):
    y, x, height_half, width_half = (int(center_y*MODEL_IMAGE_HEIGHT), int(center_x*MODEL_IMAGE_HEIGHT), int(scale_y*MODEL_IMAGE_WIDTH), int(scale_x*MODEL_IMAGE_WIDTH))
    tl_y = y - height_half
    tl_x = x - width_half
    br_y = y + height_half
    br_x = x + width_half
    
    # not outside of the image bounds
    tl_y = min(max(tl_y, 0), MODEL_IMAGE_HEIGHT-1)
    tl_x = min(max(tl_x, 0), MODEL_IMAGE_WIDTH-1)
    br_y = min(max(br_y, 0), MODEL_IMAGE_HEIGHT-1)
    br_x = min(max(br_x, 0), MODEL_IMAGE_WIDTH-1)
    
    # at least an area of 1 pixel
    if tl_y == br_y:
        if tl_y > 0:
            tl_y -= 1
        else:
            br_y += 1
    if tl_x == br_x:
        if tl_x > 0:
            tl_x -= 1
        else:
            br_x += 1
    
    return tl_y, tl_x, br_y, br_x

def visualize_rectangle(image, min_x, max_x, min_y, max_y, color_tuple, channel_is_first_axis=False):
    if len(color_tuple) > 0 and GRAYSCALE:
        print("[WARNING] got 3-channel color tuple in visualize_rectangle(), but grayscale is active.", color_tuple)
        color_tuple = 255
    
    if channel_is_first_axis:
        image = np.rollaxis(image, 0, 3)
    
    for x in range(min_x, max_x+1):
        image[min_y, x, ...] = color_tuple
        image[max_y, x, ...] = color_tuple
    for y in range(min_y, max_y+1):
        image[y, min_x, ...] = color_tuple
        image[y, max_x, ...] = color_tuple
    
    if channel_is_first_axis:
        image = np.rollaxis(image, 2, 0)
    
    return image

def predict_on_images(model, images):
    y_preds_model = model.predict(images, batch_size=128)
    
    y_preds = []
    for i, y_pred in enumerate(y_preds_model):
        #image_height = original_images[i].shape[0]
        #image_width = original_images[i].shape[1]
        tl_y, tl_x, br_y, br_x = center_scale_to_pixels(images[i], y_pred[0], y_pred[1], y_pred[2], y_pred[3])
        
        y_preds.append((tl_y, tl_x, br_y, br_x))
    return y_preds

if __name__ == "__main__":
    main()
