import gin
import logging
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

def create_record(img_dir, csv_dir, filename_record):
    # copied from tensorflow.org (wiki)
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # copied from tensorflow.org (wiki)
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def preprocess_image(image_path):
        img = tf.io.read_file(image_path)
        # directly decode from format to uint8 scaling
        img = tf.io.decode_image(img, dtype=tf.uint8)
        # get the image to desired size
        img = tf.image.resize(img, size=(256, 256))
        # cast again to uint8
        img = tf.cast(img, tf.uint8)
        # encode img again
        img = tf.io.encode_jpeg(img, quality=99)
        return img

    logging.info('Generating new tf_record....')

    with tf.io.TFRecordWriter(filename_record) as writer:
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

        # read images -> preprocess -> save to tfrecord
        for image in df_sum["Image name"]:
            image_path = os.path.join(img_dir, image + '.jpg')
            img = preprocess_image(image_path)
            label = df_sum.loc[df_sum['Image name'] == image, 'dr'].values[0]
            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(img)}

            # Create sample for record
            sample = tf.train.Example(features=tf.train.Features(feature=feature))
            # Add sample to open record
            writer.write(sample.SerializeToString())


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

def read_record(record_filename, train_val_split=1.0):
    input_dataset = tf.data.TFRecordDataset(record_filename)
    feature_description = {'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                           'image': tf.io.FixedLenFeature([], tf.string, default_value='')}
    def _parse_function(example):
        return tf.io.parse_single_example(example, feature_description)

    parsed_dataset = input_dataset.map(_parse_function)

    if train_val_split < 1.0:
        count = 0
        for _ in parsed_dataset:
            count += 1
        # shuffling, to prevent bias after splitting in train and val
        dataset = parsed_dataset.map(lambda x: (tf.io.decode_jpeg(x['image']), x['label'])).shuffle(
            buffer_size=count)
        train_dataset = dataset.take(int(count*train_val_split))
        val_dataset = dataset.skip(int(count*train_val_split))
        return train_dataset, val_dataset

    else:
        dataset = parsed_dataset.map(lambda x: (tf.io.decode_jpeg(x['image']), x['label']))
        return dataset

@gin.configurable
def load(load_record, img_dir, csv_dir):
    if load_record:
        logging.info('Loading dataset from tensorflow records....')

        train_record_filename = './input_pipeline/records/train.tfrecord'
        test_record_filename = './input_pipeline/records/test.tfrecord'

        train_set, val_set = read_record(record_filename=train_record_filename, train_val_split=0.8)
        test_set = read_record(record_filename=test_record_filename)
    else:
        logging.info('Creating new tensorflow records from dataset....')

        # Read both train and test set separately
        train_img_dir = img_dir + '/train'
        test_img_dir = img_dir + '/test'
        train_csv_dir = csv_dir + '/labels/train.csv'
        test_csv_dir = csv_dir + '/labels/test.csv'
        train_record_filename = './input_pipeline/records/train.tfrecord'
        test_record_filename = './input_pipeline/records/test.tfrecord'
        create_record(img_dir=train_img_dir, csv_dir=train_csv_dir, filename_record=train_record_filename)
        create_record(img_dir=test_img_dir, csv_dir=test_csv_dir, filename_record=test_record_filename)

        logging.info('Created new record files from dataset....')

        train_set, val_set = read_record(record_filename=train_record_filename, train_val_split=0.8)
        test_set = read_record(record_filename=test_record_filename)

    return train_set, val_set, test_set

# TODO: Preprocessing of images (Resizing/Scaling/Cut because of black rounding)
# TODO: Provide a even distribution between classes (Resampling?)
# TODO: Data augmentation (fliping etc.)