import gin
import tensorflow as tf
import logging
import pandas as pd

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
    logging.info('Samples of No DR: {}'.format(len(df_no_dr)))
    logging.info('Samples of DR: {}'.format(len(df_dr)))

    # define a new column/feature for dr
    df_dr['dr'] = 1
    df_no_dr['dr'] = 0
    df_sum = pd.concat([df_no_dr, df_dr])