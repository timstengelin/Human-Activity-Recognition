import os

import gin
import tensorflow as tf
import logging
import pandas as pd
import numpy as np
import sys

@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    return image, label

def augment(image, label):
    """Data augmentation"""

    return image, label

@gin.configurable
def create_dataset(img_dir, csv_dir, oversampling=True):
    logging.info('Generating new dataset....')

    def preprocess_image(image_path):
        img = tf.io.read_file(image_path)
        # directly decode from format to uint8 scaling
        img = tf.io.decode_image(img, dtype=tf.uint8)
        # get the image to desired size
        img = tf.image.resize(img, size=(256, 256))
        return img

    # read image + label (retinopathy grade) from csv file
    df = pd.read_csv(csv_dir)

    # separate teh different grades into separate dataframes
    df_grade_rg_0 = df[df['Retinopathy grade'] == 0]
    df_grade_rg_1 = df[df['Retinopathy grade'] == 1]
    df_grade_rg_2 = df[df['Retinopathy grade'] == 2]
    df_grade_rg_3 = df[df['Retinopathy grade'] == 3]
    df_grade_rg_4 = df[df['Retinopathy grade'] == 4]

    # create two different dataframes for classification of no dr or dr
    df_no_dr = pd.concat([df_grade_rg_0, df_grade_rg_1])
    df_dr = pd.concat([df_grade_rg_2, df_grade_rg_3, df_grade_rg_4])

    # visualize the number of available samples per class
    logging.info('Samples with No DR: {}'.format(len(df_no_dr)))
    logging.info('Samples with DR: {}'.format(len(df_dr)))

    # define a new column/feature for dr
    df_dr['dr'] = 1
    df_no_dr['dr'] = 0
    df_sum = pd.concat([df_no_dr, df_dr])

    data = []
    for image in df_sum["Image name"]:
        image_path = os.path.join(img_dir, image + '.jpg')
        img = preprocess_image(image_path)
        label = df_sum.loc[df_sum['Image name'] == image, 'dr'].values[0]
        sample = [img, label]
        data.append(sample)
    np_data = np.asarray(data, dtype=object)
    df_data = pd.DataFrame({'Image': np_data[:, 0], 'Label': np_data[:, 1]})